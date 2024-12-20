from typing import List, Tuple
import os
from pathlib import Path
import copy
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw


from xmot.logger import Logger
from xmot.digraph.node import Node
from xmot.digraph.trajectory import Trajectory
from xmot.digraph.particle import Particle
import xmot.digraph.commons as commons
import xmot.digraph.utils as utils
from xmot.mot.utils import cen2cor, draw_dashed_rectangle
import xmot.analyzer.shapeDetector as shape_detector

class Digraph:
    """
        Bass type of directed graph, composed by nodes and directed trajectories.

        Attributes:
            trajs : [Trajectory] List of particles trajectories.
            nodes : [Node]       List of nodes. I.e. events in the video
            ptcls : [Particles]  List of particles. Not necessary, but useful for
                                 accessing all particles and performing particle-wise
                                 detection and anlysis, like bubble and shape detection.
    """

    def __init__(self, nodes: List[Node] = [], trajs: List[Trajectory] = [],
                 ptcls: List[Particle] = [],):
        self.nodes = nodes
        self.trajs = trajs
        self.ptcls = ptcls

        self.__in_nodes = {}  # "node: [nodes]" pair. The value list contains nodes that
                              # have edges coming towards the key node.
        self.__out_nodes = {} # "node: [nodes]" pair. The value list contains nodes that
                              # the key node has outgoing edges pointing to.

    def add_video(self, particles: List[Particle]):
        """Load particles of video into digraph.

        Args:
            data: List of all particles identified in all frames of a video.
        """
        for p in particles:
            if p not in self.ptcls:
                self.ptcls.append(p)
            for traj in self.trajs:
                if p.get_id() == traj.id:
                    # Note: trajectory attributes: start_node, end_node and kalmanfilter
                    # are still None.
                    #traj.add_particle(p)
                    traj.append_particle(p) # At initialization stage, we can simply append
                                            # the particle without any checks.
                    break
            else:
                # This particle doesn't belong to any existing trajs. Create a new traj.
                traj = Trajectory(id = p.id, ptcls = [p])
                #node = Node()
                #traj.set_start_node(node)
                #node.add_out_traj(traj) # id of the underlying particle will be
                                        # automatically added to the node.
                self.trajs.append(traj)
                #self.nodes.append(node)

        # Post-processing:
        # This operation was meant to dealt with flickering issues. But it causes more
        # error messages than it can solve for the flickering problem. In fact, for particles
        # that have flickering problems, they're not static and could move really far away
        # itself in directly preceding frame. So the "close in time and space" criterion of
        # merging might not work. (Or perhaps we can keep the trajectories that cannot merge).
        #self.__merge_short_trajs()

        if len(self.trajs) == 0: # No particles detected in the video
            return

        # Kalman filter has an allowance for particles to be undetected for a few frames before
        # totally removing it. This leads to two fictious particles with no contours at the end
        # of each Kalman filter trajectory. Check their existance and remove them.
        for traj in self.trajs:
            removed_ptcls = traj.remove_no_contour_particles_at_end()
            for p in removed_ptcls:
                self.ptcls.remove(p)

        # Check trajectories and break up the ones that should be separated
        # After this sanity check, trajectory id does not necessarily align with the chronological
        # order of start time.
        max_id = max([t.get_id() for t in self.trajs])
        for traj in self.trajs.copy(): # Avoid index shifting bug
            new_traj = Trajectory.break_trajectory(traj)
            if new_traj is not None:
                self.trajs.append(new_traj)
                new_traj.set_id(max_id + 1) # set id for trajectory and underlying particles
                max_id += 1

        # Attach a start node and end node to each of the trajectory
        for traj in self.trajs:
            start_node = Node(in_trajs = None, out_trajs = [traj])
            end_node = Node(in_trajs = [traj], out_trajs = None)
            traj.set_start_node(start_node)
            traj.set_end_node(end_node)
            self.nodes += [start_node, end_node]

        # Event detection and merge nodes.
        self.__detect_events2()

    def __merge_short_trajs(self):
        """
        This function is to resolve the flickering issue in which the same particles are
        repeatedly picked up by detection algorithm and then dropped off in next few frames.

        If a trajectory only exists for less than 5 time_frames, and don't exit the video.
        Glue it to nearest trajectories that are both close in time and space.
        """
        short_trajs: List[Trajectory] = []
        long_trajs: List[Trajectory] = []
        for traj in self.trajs:
            if traj.get_life_time() <= 5:
                short_trajs.append(traj)    # shallow copy
            else:
                long_trajs.append(traj)     # shallow copy
        for st in short_trajs:
            for lt in long_trajs:
                dist = utils.traj_distance(st, lt)
                if dist < utils.CLOSE_IN_SPACE:
                    ptcls = copy.deepcopy(st.get_particles())
                    Logger.detail("Try to merge trajectory {:d} into {:d}".format(st.id, lt.id))
                    if not lt.merge_particles(ptcls):
                        # Failed to merge becuase long_traj already has particles at time frames
                        # of the short trajectory
                        Logger.error("Fail to merge trajectory {:d} into {:d}".format(st.id, lt.id))
                        continue
                    self.trajs.remove(st)

                    # This function is only used during digraph initialization and at the stage
                    # before any nodes are created. So the following checks currently don't have
                    # any effect. But they're good to have in case in the future this merge
                    # function is called after digraph initializaion.
                    if st.start_node in self.nodes:
                        self.nodes.remove(st.start_node)
                    if st.end_node in self.nodes:
                        self.nodes.remove(st.end_node)
                    break

    def __detect_events(self):
        """
        (Deprecated) Loop repeatively to merge nodes into events: collision and micro-explosion.
        """
        # micro-explosion
        sorted_trajs = []
        event_times = set([])
        for traj in self.trajs:
            if traj.get_start_time() == 0:      # Cannot check events happen exactly at the beginning of the video.
                continue
            sorted_trajs.append(traj)
        sorted_trajs.sort(key=lambda t: t.get_start_time())
        trajs_start_merged = []     # Trajectories whose start node has been merged with
                                    # a start node of other trajs earlier in time.
        trajs_end_merged = []       # Trajectories whose end node has been merged with trajs later in time
        for i in range(len(sorted_trajs)):
            t_i = sorted_trajs[i]
            event_time = t_i.get_start_time()
            # Step 1: Check the start nodes of trajectories that start later than this trajectory.
            if t_i in trajs_start_merged:
                continue
            for j in range(i + 1, len(sorted_trajs)):
                t_j = sorted_trajs[j]
                if t_j in trajs_start_merged:
                    continue
                if t_j.get_start_time() - event_time <= utils.CLOSE_IN_TIME and \
                   t_j.get_position_by_time(event_time) != None:
                    dist = utils.distance(t_i.get_position_by_time(event_time),
                                          t_j.get_position_by_time(event_time))
                    if dist <= utils.CLOSE_IN_SPACE:
                        # Close in both space and time, merge.
                        Logger.debug("Event detected: merge start nodes of " \
                                     "trajectories: {:d} {:d}".format(t_i.get_id(), t_j.get_id()))
                        trajs_start_merged.append(t_j)

                        # TODO: collect the following ops into a function. e.g. merge_nodes()
                        temp_node = t_j.get_start_node()
                        self.nodes.remove(temp_node)
                        # TODO: remove references from self.__in_nodes and self.__out_nodes.
                        # Or recreate them after the merge complete.
                        # self.__out_nodes.remove(t_j.get_start_node())
                        t_j.set_start_node(t_i.get_start_node())
                        for t in temp_node.get_out_trajs():
                            t_i.get_start_node().add_out_traj(t)
                        for t in temp_node.get_in_trajs():
                            t_i.get_start_node().add_in_traj(t)
                        event_times.add(event_time)
            # Step 2: Check the end nodes of trajectories that start earlier than this trajectory.
            for k in range(0, i):  # Only trajectories start earlier than t_i can finish before t_i starts.
                t_k = sorted_trajs[k]
                if t_k in trajs_end_merged:
                    # The end node of t_k has been merged with ealier start nodes of other trajectory.
                    # Don't need to be tested with
                    continue
                if abs(t_k.get_end_time() - event_time) <= utils.CLOSE_IN_TIME and \
                   t_k.get_position_by_time(event_time) != None:
                    dist = utils.distance(t_i.get_position_by_time(event_time),
                                          t_k.get_position_by_time(event_time))
                    if dist <= utils.CLOSE_IN_SPACE:
                        Logger.debug("Event detected: merge start node and end node of "\
                                     "trajectories: {:d} {:d}".format(t_i.get_id(), t_k.get_id()))
                        trajs_end_merged.append(t_k)
                        temp_node = t_k.get_end_node()
                        self.nodes.remove(temp_node)
                        t_k.set_end_node(t_i.get_start_node())
                        for t in temp_node.get_in_trajs():
                            t_i.get_start_node().add_in_traj(t)
                        for t in temp_node.get_out_trajs():
                            t_i.get_start_node().add_out_traj(t)
                        event_times.add(event_time)
                        # TODO: deal with self.__in_nodes and self.__out_nodes
        # TODO: Change event_times to map of time and positions.
        # TODO: Split trajectories that pass the event position at exactly the event time.
        #       They could be new particles but accidently inherited the old trajectory.
        pass

    def __detect_events2(self):
        """
        A simpler function to detect events and merge nodes. It parewisely compare both the spatial
        and temporal distances of pairs of nodes. When a pair of nodes are determined to be close,
        merge the node with a later start time into the node with an earlier start time (via
        iterating trajectories in chronological order).

        The rule must be obeyed for merging: A start node of a trajecotry must start earlier in time
        than its end node.

        This rule implies two practical checks for short trajectories:
        1. For a trajectory that starts earlier in time, its start node cannot be merged with
           end nodes of a trajectory that starts later in time.
        2. The end node of a trajectory A can't be merged with the start node of a trajectory B whose
           start node starts later in time but has been merged with the start node of A.

        """
        self.trajs.sort(key=lambda t: t.get_start_time())

        for i in range(0, len(self.trajs)):
            # Can't detect event at the start of a video. <TODO> Is this true?
            # if self.trajs[i].get_start_time() == 0:
            #     continue
            for j in range(i + 1, len(self.trajs)):
                # t_j always starts no ealier than t_i.
                t_i = self.trajs[i]
                t_j = self.trajs[j]

                # Case 1. Check the start node of t_i with start node of t_j:
                event_time = t_i.get_start_time()
                if t_j.get_start_time() - event_time <= utils.CLOSE_IN_TIME and \
                   t_j.get_position_by_time(event_time) != None:
                    dist = utils.distance(t_i.get_position_by_time(event_time),
                                          t_j.get_position_by_time(event_time))
                    if dist <= utils.CLOSE_IN_SPACE:
                        # Close in both space and time, merge.
                        Logger.debug("Event detected: merge start nodes of " \
                                     "trajectories: {:d} {:d}".format(t_i.get_id(), t_j.get_id()))

                        # TODO: remove references from self.__in_nodes and self.__out_nodes.
                        # Or recreate them after the merge complete.
                        # self.__out_nodes.remove(t_j.get_start_node())

                        # The nodes to be merged are not already the same. Otherwise, there'll
                        # be an infinite loop.
                        _src = t_j.get_start_node()
                        _dest = t_i.get_start_node() # "dest" in the sense to merge trajs into.
                        if _src != _dest:
                            Digraph.__merge_node(src=_src, dest=_dest)
                            self.nodes.remove(_src)

                # 2. Check the end node of t_i with start node of t_j:
                event_time = t_i.get_end_time()

                # Start node of t_j can't already be merged with the start node of t_i. Otherwise,
                # do not need to check closeness in time and space. Even if they are close, we can't
                # merge.
                if t_j.get_start_node() != t_i.get_start_node() and \
                   abs(t_j.get_start_time() - event_time) <= utils.CLOSE_IN_TIME and \
                   t_j.get_position_by_time(event_time) != None:
                    dist = utils.distance(t_i.get_position_by_time(event_time),
                                          t_j.get_position_by_time(event_time))
                    if dist <= utils.CLOSE_IN_SPACE:
                        Logger.debug(f"Event detected: merge start node of {t_i.get_id()} and "
                                     f"end node of {t_j.get_id()}")

                        _src = t_j.get_start_node()
                        _dest = t_i.get_end_node() # "dest" in the sense to merge trajs into.
                        if _src != _dest:
                            Digraph.__merge_node(src=_src, dest=_dest)
                            self.nodes.remove(_src)

                # 3. Check the end node of t_i with end node of t_j:
                event_time = t_i.get_end_time()

                # End node of t_i has not already been merged with start node of t_j.
                if t_i.get_end_node() != t_j.get_start_node() and \
                   abs(t_j.get_end_time() - event_time) <= utils.CLOSE_IN_TIME and \
                   t_j.get_position_by_time(event_time) != None:
                    dist = utils.distance(t_i.get_position_by_time(event_time),
                                          t_j.get_position_by_time(event_time))
                    if dist <= utils.CLOSE_IN_SPACE:
                        Logger.debug(f"Event detected: merge end node of {t_i.get_id()} and "
                                     f"end node of {t_j.get_id()}")

                        _src = t_j.get_end_node()
                        _dest = t_i.get_end_node() # "dest" in the sense to merge trajs into.
                        if _src != _dest:
                            Digraph.__merge_node(src=_src, dest=_dest)
                            self.nodes.remove(_src)

    def __merge_node(src, dest):
        for traj in src.get_out_trajs():
            dest.add_out_traj(traj)
            traj.set_start_node(dest)
        for traj in src.get_in_trajs():
            dest.add_in_traj(traj)
            traj.set_end_node(dest)


    def add_node(self, node: Node):
        if node in self.__in_nodes and node in self.__out_nodes:
            Logger.debug("Node {:s} already in the list".format(node.to_str()))
            return

        if node not in self.__in_nodes:
            self.__in_nodes[node] = []

        if node not in self.__out_nodes:
            self.__out_nodes[node] = []

    def add_edge(self, start, end):
        if start not in self.__in_nodes or start not in self.__out_nodes:
            Logger.debug("Trying to add an edge for a non-existing node {:s}".format(start.to_str()))
            self.add_node(start)

        if end not in self.__in_nodes or end not in self.__out_nodes:
            Logger.debug("Trying to add an edge for a non-existing node {:s}".format(start.to_str()))
            self.add_node(end)

        self.__in_nodes[end].append(start)
        self.__out_nodes[start].append(end)

    def has_node(self, node):
        return node in self.__in_nodes or node in self.__out_nodes

    def has_edge(self, start, end):
        if not self.has_node(start) or not self.has_node(end):
            return False

        for node in self.__out_nodes[start]:
            if node == end:
                return True

        return False

    def get_particles(self):
        """A window for accessing directly all identified particles in the digraph."""
        return self.ptcls

    #def del_node(self, node):
    #    """
    #        Delete node from the digraph.
    #
    #        Note that all edges related to the node are deleted as well.
    #    """
    #    #del self.__in_nodes[node]
    #    #del self.__out_nodes[node]
    #    #for n in self.__in_nodes:
    #    #    self.__in_nodes[n].remove(node)
    #    #for n in self.__out_nodes:
    #    #    self.__out_nodes[n].remove(node)
    #    pass
    #
    #def del_edge(self, start, end):
    #    """
    #        Delete edge from the digraph.
    #
    #        If the deleted edge is the only edge for the nodes start and end,
    #        the two nodes are not deleted. (<todo> maybe we should also delete the
    #        nodes that have no edges connecting to them).
    #    """
    #    #self.__in_nodes[end].remove(start)
    #    #self.__out_nodes[start].remove(end)
    #    pass
    #
    #def reverse(self):
    #    """
    #        Reverse the direction of all edges.
    #    """
    #    #temp = self.__in_nodes
    #    #self.__in_nodes = self.__out_nodes
    #    #self.__out_nodes = temp
    #    pass


    def draw(self, dest, write_img=True, draw_id=False, draw_shape=True, start_frame=-1, end_frame=-1) -> List[Image.Image]:
        """
            Draw trajectories and nodes in pictures. One picture for each time frame.

            Args:
                dest      : String  Path of the folder to put the drawings.
                write_img : Boolean Flag controlling whether to write images.
                draw_id   : Boolean Whether draw particle id on top right corner of bbox.
                draw_shape: Boolean Whether write shape of particle on top right corner.
                start_frame: Index of first frame from which to draw. If -1, use the smallest frame index having a particle.
                end_frame : Index of last frame to end drawing. If -1, use the largest frame index having a particle.
            Returns:
                A list of Image objects representing reproduced frames from the digraph
                representation.
        """
        images = []
        if write_img:
            os.makedirs(dest, exist_ok=True)

        # Group entities according to associated time_frame and then draw
        # frame by frame.
        start_frame_digraph, end_frame_digraph = self.get_time_frames()
        if start_frame == -1:
            start_frame = start_frame_digraph
        if end_frame == -1:
            end_frame == end_frame_digraph
        dict_ptcls = {} # Dict{int: List[Particle]}
        dict_trajs = {}
        dict_nodes = {}
        for traj in self.trajs:
            for t in range(traj.get_start_time(), traj.get_end_time() + 1):
                if t in dict_trajs:
                    dict_trajs[t].append(traj)
                else:
                    dict_trajs[t] = [traj]

            for p in traj.get_particles():
                if p.time_frame in dict_ptcls:
                    dict_ptcls[p.time_frame].append(p)
                else:
                    dict_ptcls[p.time_frame] = [p]

        for node in self.nodes:
            if node.get_start_time() in dict_nodes:
                dict_nodes[node.get_start_time()].append(node)
            else:
                dict_nodes[node.get_start_time()] = [node]

        # Drawing
        for t in range(start_frame, end_frame + 1):
            im = Image.new("RGB", commons.PIC_DIMENSION, (180, 180, 180))
            draw = ImageDraw.Draw(im)

            # Mark the event in the videoframes of its happenning.
            if t in dict_nodes:
                for node in dict_nodes[t]:
                    bbox = node.get_bbox_node()
                    if node.get_type() == "start":
                        color = (255, 0, 0, 125) # translucent, red
                    elif node.get_type() == "end":
                        color = (0, 0, 255, 125) # translucent, blue
                    else:
                        color = (0, 255, 0, 125) # translucent, green
                    #draw.rectangle(utils.torch_bbox_to_coordinates(bbox), fill=color)
            if t in dict_ptcls:
                for p in dict_ptcls[t]:
                    #bbox = p.bbox
                    #bbox = [10, 10] # For testing.
                    # For now, assume the p.position is the center of the bbox.
                    xy = [(p.position[0], p.position[1]),
                          (p.position[0] + p.bbox[0], p.position[1] + p.bbox[1])]
                    draw.rectangle(xy, outline=(50, 50, 50), width = 2) # dark gray
                    if draw_id:
                        text_xy = (p.position[0] + p.bbox[0] + 5, p.position[1] - 15)
                        # Id number in red.
                        draw.text(text_xy, str(p.get_id()), (255, 0, 0), stroke_width=2)
                    if draw_shape:
                        if p.get_shape() == "":
                            Logger.debug("Want to draw shape for particle, but missing shape information." \
                                         "id {:d} time-frame {:d}".format(p.get_id(), p.get_time_frame()))
                        else:
                            text_xy = (p.position[0] + p.bbox[0] + 5, p.position[1] - 5)
                            # Id number in red.
                            shape = p.get_shape()[0].upper()
                            draw.text(text_xy, str(shape), (255, 0, 0), stroke_width=2)
            if write_img:
                im.save("{:s}/reproduced_{:d}.png".format(dest, t)) # JPG doesn't support alpha
            images.append(im)
        return images

    def draw_overlay(self, dest, write_img, draw_id=False):
        """
        Similar to what draw() does, except that particles of all frames are drawn on the same
        picture, thus "overlay".

        Args:
            dest      : String  Path of the folder to put the drawings.
            write_img : Boolean Flag controlling whether to write images.
        Returns:
            A list of Image objects representing reproduced frames from the digraph
            representation.
        """
        images = []
        im = Image.new("RGBA", commons.PIC_DIMENSION, (180, 180, 180, 255)) # solid gray
        draw = ImageDraw.Draw(im)
        if write_img:
            os.makedirs(dest, exist_ok=True)
        start_frame, end_frame = self.get_time_frames()
        dict_ptcls = {} # Dict{int: List[Particle]}
        dict_trajs = {}
        dict_nodes = {}
        for traj in self.trajs:
            for t in range(traj.get_start_time(), traj.get_end_time() + 1):
                if t in dict_trajs:
                    dict_trajs[t].append(traj)
                else:
                    dict_trajs[t] = [traj]

            for p in traj.get_particles():
                if p.time_frame in dict_ptcls:
                    dict_ptcls[p.time_frame].append(p)
                else:
                    dict_ptcls[p.time_frame] = [p]

        for node in self.nodes:
            if node.get_start_time() in dict_nodes:
                dict_nodes[node.get_start_time()].append(node)
            else:
                dict_nodes[node.get_start_time()] = [node]

        for t in range(start_frame, end_frame + 1):
            if t in dict_nodes:
                for node in dict_nodes[t]:
                    bbox = node.get_bbox_node()
                    if node.get_type() == "start":
                        color = (255, 0, 0, 125) # translucent, red
                    elif node.get_type() == "end":
                        color = (0, 0, 255, 125) # translucent, blue
                    else:
                        color = (0, 255, 0, 125) # translucent, green
                    draw.rectangle(utils.torch_bbox_to_coordinates(bbox), fill=color)
            if t in dict_ptcls:
                for p in dict_ptcls[t]:
                    #bbox = p.bbox
                    #bbox = [10, 10] # For testing.
                    # For now, assume the p.position is the center of the bbox.
                    xy = [(p.position[0], p.position[1]),
                          (p.position[0] + p.bbox[0], p.position[1] + p.bbox[1])]
                    draw.rectangle(xy, outline=(50, 50, 50, 255), width = 2) # solid dark gray
            if write_img:
                im.save("{:s}/reproduced_{:d}.png".format(dest, t))
            images.append(im.copy())
        return images

    def draw_line_format(self, dest, write_img, draw_id=False):
        """
            Draw trajectories and nodes in pictures with trajectories represented by
            piecewise lines connecting the underlying particle at different time frame and with
            nodes represented by squares.

            For reliability, draw only nodes whose life times are greater than 5.
            <TODO> After we have more reliable identification by neural networks, we should remove
            this constraint.

            Args:
                dest      : String  Path of the folder to put the drawings.
                write_img : Boolean Flag controlling whether to write images.
            Returns:
                A list of Image objects representing reproduced frames from the digraph
                representation.
        """
        images = []
        im = Image.new("RGBA", commons.PIC_DIMENSION, (180, 180, 180, 255))# solid gray
        draw = ImageDraw.Draw(im)
        if write_img:
            os.makedirs(dest, exist_ok=True)
        start_frame, end_frame = self.get_time_frames()
        dict_ptcls = {} # Dict{int: List[Particle]}
        dict_trajs = {}
        dict_nodes = {}
        for traj in self.trajs:
            for t in range(traj.get_start_time(), traj.get_end_time() + 1):
                if t in dict_trajs:
                    dict_trajs[t].append(traj)
                else:
                    dict_trajs[t] = [traj]

            for p in traj.get_particles():
                if p.time_frame in dict_ptcls:
                    dict_ptcls[p.time_frame].append(p)
                else:
                    dict_ptcls[p.time_frame] = [p]

        for node in self.nodes:
            if node.get_start_time() in dict_nodes:
                dict_nodes[node.get_start_time()].append(node)
            else:
                dict_nodes[node.get_start_time()] = [node]

        for t in range(start_frame, end_frame + 1):
            if t in dict_nodes:
                for node in dict_nodes[t]:
                    # <TODO> remove this ad hoc constraint
                    # Don't draw nodes that are associated with trajectories that lives shorter
                    # than 5 time frames.
                    for traj in node.in_trajs + node.out_trajs:
                        if traj.get_life_time() > 5:
                            break
                    else:
                        continue # All trajectories are shorter than 5 time frames.
                    bbox = node.get_bbox_node()
                    if node.get_type() == "start":
                        color = (255, 0, 0, 125) # translucent, red
                    elif node.get_type() == "end":
                        color = (0, 0, 255, 125) # translucent, blue
                    else:
                        color = (0, 255, 0, 125) # translucent, green
                    draw.rectangle(utils.torch_bbox_to_coordinates(bbox), fill=color)
            if t in dict_trajs:
                # Grow trajectories that exist at time t.
                for traj in dict_trajs[t]:
                    # <TODO> remove this ad hoc constraint
                    if traj.get_life_time() <= 5:
                        continue
                    if t != traj.get_start_time():
                        # p is not the first particle in this trajectory
                        p = traj.get_particle(t)
                        if p is None: continue
                        p2 = None
                        for t2 in range(t - 1, traj.get_start_time() - 1, -1):
                            p2 = traj.get_particle(t2)
                            if p2 != None: break

                        if p2 is None:
                            Logger.error("Something is wrong with this trajectory. " +
                                         "No particle exists before this frame which is not " +
                                         "the starting frame. {:d} {:d}".format(traj.id, t))
                            continue # go to next trajectory
                        xy = [tuple(p2.get_position()), tuple(p.get_position())]
                        draw.line(xy, fill=(50, 50, 50, 255), width=2) # solid dark gray
            if write_img:
                im.save("{:s}/reproduced_{:d}.png".format(dest, t))
            images.append(im.copy())
        return images

    def get_time_frames(self) -> Tuple[int]:
        """
            Find range of frames for the video represented by the digraph.

            The start frame is always 0. The end frame if the largest time_frame
            of all trajectories.
        """
        start_frame, end_frame = 0, 0
        for traj in self.trajs:
            end_frame = traj.get_end_time() if traj.get_end_time() > end_frame else end_frame
        return (start_frame, end_frame)

    def detect_particle_shapes(self, video=None, images=None):
        """
        Args:
            video   String  Path to the original video file.
        """
        if images is None and video is None:
            Logger.error("SHAPE DETECTION: Need to provide path to the video or the extracted images")
            return

        # If both are given, use the given images. Don't load again.
        if images is None and video is not None:
            images = utils.extract_images(video, to_gray=True)

        for p in self.ptcls:
            shape = shape_detector.detect_shape(p, images[p.get_time_frame()])
            p.set_shape(shape)

    def draw_events(self, images, outdir, type: str = None, prefix: str = None):
        """
        Highlight nodes whose type is not 'start' or 'end' and the involving trajectories.
        Draw the bbox of the nodes, contours and IDs of the particles in the trajectories.

        Args:
            images: The original images. Overlay the events on them.
            outdir: The output folder for the events. Each event will has its subfolder.
            type: Highlight only the given type of events.
            prefix: Prefix of saved image. E.g. use video ID with material information.
        """
        # TODO: ID the events and store them as an attribute of the digraph object, rather than
        # temporarily in this function.
        counts = {}
        outdir_path = Path(outdir)
        for node in self.nodes:
            event_type = node.get_type()
            if type is not None and event_type != type:
                continue

            # Only skip the 'start' and 'end' type.
            if event_type == 'start' or event_type == 'end':
                continue

            # Draw the event
            event_id = counts.get(event_type, 0)
            event_dir = outdir_path.joinpath(f'{event_type}_{event_id}')
            event_dir.mkdir(parents=True, exist_ok=True)
            prefix = prefix if prefix is not None else f'{event_type}_{event_id}'
            event_start_time = node.get_start_time()
            event_end_time = node.get_end_time()

            # Pre-event: Draw 5 frame pre event
            for time in range(max(event_start_time - 5, 0), event_start_time):
                # opencv draw objects inplace. Use a copy to not change the original images.
                img_result = copy.deepcopy(images[time])
                img_result = cv.cvtColor(img_result, cv.COLOR_GRAY2BGR)
                # Write the time frame on the upper-left corner in green.
                cv.putText(
                    img_result, str(time), np.array((30, 30)), cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=(0, 255, 0), thickness = 1, lineType=cv.LINE_AA
                )

                has_traj = False # Flag to control whether trajectories exist at this time frame
                                 # pre-event.
                for traj in node.get_in_trajs(): # Only incoming trajs exist pre event.
                    if traj.get_start_time() > time:
                        continue

                    img_result = Digraph._draw_particle_with_id(traj, time, img_result, in_traj=True)
                    has_traj = True

                if has_traj:
                    # Only save image when trajectories exist at this time frame before the event.
                    cv.imwrite(str(event_dir.joinpath(f"{prefix}_{time}.png")), img_result)

            # During the event
            for time in range(event_start_time, event_end_time + 1):
                img_result = copy.deepcopy(images[time])
                img_result = cv.cvtColor(img_result, cv.COLOR_GRAY2BGR)

                cv.putText(
                    img_result, str(time), np.array((30, 30)), cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=(0, 255, 0), thickness = 1, lineType=cv.LINE_AA
                )

                # Draw the bbox enclosing the event in green.
                cv.rectangle(
                    img_result, *utils.torch_bbox_to_coordinates(node.get_bbox_node()),
                    color=(0, 255, 0), thickness = 2
                )

                for traj in node.get_in_trajs() + node.get_out_trajs():
                    img_result = Digraph._draw_particle_with_id(traj, time, img_result, node.is_in_traj(traj))

                # Save the image
                cv.imwrite(str(event_dir.joinpath(f"{prefix}_{time}.png")), img_result)

            # Post-event: Draw 5 frames after the event
            for time in range(event_end_time + 1, min(event_end_time + 6, len(images))):
                img_result = copy.deepcopy(images[time])
                img_result = cv.cvtColor(img_result, cv.COLOR_GRAY2BGR)
                cv.putText(
                    img_result, str(time), np.array((30, 30)), cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=(0, 255, 0), thickness = 1, lineType=cv.LINE_AA
                )

                has_traj = False # Flag to control whether trajectories exist at this time frame
                                 # pre-event.
                for traj in node.get_out_trajs(): # Only outgoing trajs exist pre event.
                    if traj.get_end_time() < time:
                        continue

                    img_result = Digraph._draw_particle_with_id(traj, time, img_result, in_traj=False)
                    has_traj = True

                if has_traj:
                    # Only save image when trajectories exist at this time frame before the event.
                    cv.imwrite(str(event_dir.joinpath(f"{prefix}_{time}.png")), img_result)

            # Increment the event count.
            counts[event_type] = event_id + 1


    def _draw_particle_with_id(traj: Trajectory, time: str, img: np.ndarray, in_traj: bool) -> np.ndarray:
        p = traj.get_particle_by_time(time)
        if p is None or p.get_contour() is None:
            # At this given time frame, there's no particle in the trajectory or the
            # particle is inferred by Kalman filters (not detected at detection step).
            # The former case happen when the event is detected within the
            # BACK_TRACE_LIMIT.

            # Draw an estimated particle bbox with dashed lines in red.
            _position = traj.get_position_by_time(time)

            if _position is None: # trajectory is away from 'time' by more than BACK_TRACE_LIMIT
                return img

            if in_traj:
                _bbox = traj.get_end_particle().get_bbox()
            else:
                _bbox = traj.get_start_particle().get_bbox()
            _bbox_torch = cen2cor(*_position, *_bbox)
            img = draw_dashed_rectangle(
                img, *utils.torch_bbox_to_coordinates(_bbox_torch),
                color=(0, 0, 255), thickness = 1, dash_length=2, gap_length=1,
                inplace=True
            )
        else:
            # Draw particle contours (not Kalman Filter adjusted positions) in red.
            img = cv.drawContours(
                img, [p.get_contour()], -1, color=(0, 0, 255), thickness=1
            )

            _bbox_torch = p.get_contour_bbox_torch()

        # Write the particle ID (eq. to traj ID) at upper right corner of particle bbox.
        cv.putText(
            img, str(traj.get_id()),
            (int(_bbox_torch[2] + 5), int(_bbox_torch[1] - 5)),
            cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), # BGR
            thickness=1, lineType=cv.LINE_AA
        )

        return img

    def __str__(self):
        """
            Return a string representation of the directed graph. <todo>
        """
        string = ""
        for traj in self.trajs:
            string += str(traj) + os.linesep
        for node in self.nodes:
            string += str(node) + os.linesep
        for p in self.ptcls:
            string += str(p) + os.linesep
        string.strip()
        return string