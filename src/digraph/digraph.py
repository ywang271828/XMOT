from typing import List, Tuple
import os
import copy
from PIL import Image, ImageDraw

from logger import Logger
from digraph.node import Node
from digraph.trajectory import Trajectory
from digraph.particle import Particle
import digraph.commons as commons
import digraph.utils as utils

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
                            # have outgoing edges towards the key.
        self.__out_nodes = {} # "node: [nodes]" pair. The value list contains nodes that
                            # the key node have outgoing edges pointing to.

    def add_video(self, particles):
        """Load particles of video into digraph.
        
        Args:
            data: List of all particles identified in all frames of a video.
        """
        for p in particles:
            if p not in self.ptcls:
                self.ptcls.append(p)
            for traj in self.trajs:
                if p.id == traj.id:
                    # <TODO> start_node, end_node and kalmanfilter are still None.
                    traj.add_particle(p)
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
        # 1. trajectories only exists for one time_frame, and don't exit the video. Glue them
        #    to nearest trajectories.
        self.__merge_short_trajs()

        # Attach a start node and end node to each of the trajectory
        for traj in self.trajs:
            start_node = Node()
            end_node = Node()
            traj.set_start_node(start_node)
            traj.set_end_node(end_node)
            start_node.add_out_traj(traj)
            end_node.add_in_traj(traj)
            self.nodes += [start_node, end_node]

        # <TODO> Merge nodes in collisions and micro-explosions.
        self.__detect_events()
        pass

    def __merge_short_trajs(self):
        """
            If a trajectory only exists for less than 5 time_frames, and don't exit the video. 
            Glue it to nearest trajectories that are both close in time and space.
        """
        short_trajs: List[Trajectory] = []
        long_trajs: List[Trajectory] = []
        for traj in self.trajs:
            if traj.get_life_time() <= 5:
                short_trajs.append(traj)
            else:
                long_trajs.append(traj)
        for st in short_trajs:
            for lt in long_trajs:
                dist = utils.traj_distance(st, lt)
                if dist < utils.CLOSE_IN_SPACE:
                    ptcls = copy.deepcopy(st.get_particles())
                    Logger.detail("Try to merge trajectory {:d} into {:d}".format(st.id, lt.id))
                    if not lt.add_particles(ptcls, merge=True):
                        # Failed to merge becuase long_traj already has particles at time frames
                        # of the short trajectory
                        Logger.detail("Fail to merge trajectory {:d} into {:d}".format(st.id, lt.id))
                        continue
                    self.trajs.remove(st)
                    if st.start_node in self.nodes:
                        self.nodes.remove(st.start_node)
                    if st.end_node in self.nodes:
                        self.nodes.remove(st.end_node)
                    break
        
    def __detect_events():
        """
        Loop repeatively to merge nodes into events: collision and micro-explosion.
        """
        # TODO
        pass

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
            Logger.warning("Trying to add an edge for a non-existing node {:s}".format(start.to_str()))
            self.add_node(start)
        
        if end not in self.__in_nodes or end not in self.__out_nodes:
            Logger.warning("Trying to add an edge for a non-existing node {:s}".format(start.to_str()))
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

    def del_node(self, node):
        """
            Delete node from the digraph.

            Note that all edges related to the node are deleted as well.
        """
        #del self.__in_nodes[node]
        #del self.__out_nodes[node]
        #for n in self.__in_nodes:
        #    self.__in_nodes[n].remove(node)
        #for n in self.__out_nodes:
        #    self.__out_nodes[n].remove(node)
        pass

    def del_edge(self, start, end):
        """
            Delete edge from the digraph.

            If the deleted edge is the only edge for the nodes start and end,
            the two nodes are not deleted. (<todo> maybe we should also delete the
            nodes that have no edges connecting to them).
        """
        #self.__in_nodes[end].remove(start)
        #self.__out_nodes[start].remove(end)
        pass

    def reverse(self):
        """
            Reverse the direction of all edges.
        """
        #temp = self.__in_nodes
        #self.__in_nodes = self.__out_nodes
        #self.__out_nodes = temp
        pass


    def draw(self, dest, write_img=True) -> List[Image.Image]:
        """
            Draw trajectories and nodes in pictures. One picture for each time frame.

            Args:
                dest      : String  Path of the folder to put the drawings.
                write_img : Boolean Flag controlling whether to write images.
            Returns:
                A list of Image objects representing reproduced frames from the digraph
                representation.
        """
        images = []
        if write_img:
            os.makedirs(dest, exist_ok=True)

        # Group entities according to associated time_frame and then draw 
        # frame by frame.
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
        
        # Drawing
        for t in range(start_frame, end_frame + 1):
            im = Image.new("RGB", commons.PIC_DIMENSION, (180, 180, 180))
            draw = ImageDraw.Draw(im)

            # Mark the event in the videoframes of its happenning.
            if t in dict_nodes:
                for node in dict_nodes[t]:
                    bbox = node.get_bbox()
                    if node.get_type() == "start":
                        color = (255, 0, 0, 125) # translucent, red
                    elif node.get_type() == "end":
                        color = (0, 0, 255, 125) # translucent, blue
                    else:
                        color = (0, 255, 0, 125) # translucent, green
                    #draw.rectangle(bbox, fill=color)
            if t in dict_ptcls:
                for p in dict_ptcls[t]:
                    #bbox = p.bbox
                    #bbox = [10, 10] # For testing.
                    # For now, assume the p.position is the center of the bbox.
                    xy = [(p.position[0], p.position[1]),
                          (p.position[0] + p.box[0], p.position[1] + p.box[1])]
                    draw.rectangle(xy, outline=(50, 50, 50), width = 5) # dark gray
            if write_img:
                im.save("{:s}/reproduced_{:d}.png".format(dest, t)) # JPG doesn't support alpha
            images.append(im)
        return images
    
    def draw_overlay(self, dest, write_img):
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
                    bbox = node.get_bbox()
                    if node.get_type() == "start":
                        color = (255, 0, 0, 125) # translucent, red
                    elif node.get_type() == "end":
                        color = (0, 0, 255, 125) # translucent, blue
                    else:
                        color = (0, 255, 0, 125) # translucent, green
                    draw.rectangle(bbox, fill=color)
            if t in dict_ptcls:
                for p in dict_ptcls[t]:
                    #bbox = p.bbox
                    #bbox = [10, 10] # For testing.
                    # For now, assume the p.position is the center of the bbox.
                    xy = [(p.position[0], p.position[1]),
                          (p.position[0] + p.box[0], p.position[1] + p.box[1])]
                    draw.rectangle(xy, outline=(50, 50, 50, 255), width = 5) # solid dark gray
            if write_img:
                im.save("{:s}/reproduced_{:d}.png".format(dest, t))
            images.append(im.copy())
        return images

    def draw_line_format(self, dest, write_img):
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
                    bbox = node.get_bbox()
                    if node.get_type() == "start":
                        color = (255, 0, 0, 125) # translucent, red
                    elif node.get_type() == "end":
                        color = (0, 0, 255, 125) # translucent, blue
                    else:
                        color = (0, 255, 0, 125) # translucent, green
                    draw.rectangle(bbox, fill=color)
            if t in dict_trajs:
                # Grow trajectories that exist at time t.
                for traj in dict_trajs[t]:
                    # <TODO> remove this ad hoc constraint
                    if traj.get_life_time() <= 5:
                        continue
                    if t != traj.get_start_time():
                        # p is not the first particle in this trajectory
                        p = traj.get_particle(t)
                        if p == None: continue
                        p2 = None
                        for t2 in range(t - 1, traj.get_start_time() - 1, -1):
                            p2 = traj.get_particle(t2)
                            if p2 != None: break
                        
                        if p2 == None:
                            Logger.error("Something is wrong with this trajectory. " +
                                         "No particle exists before this frame which is not " +
                                         "the starting frame. {:d} {:d}".format(traj.id, t))
                            continue # go to next trajectory
                        xy = [tuple(p2.get_position()), tuple(p.get_position())]
                        draw.line(xy, fill=(50, 50, 50, 255), width=5) # solid dark gray
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
    
    def __str__(self):
        """
            Return a string representation of the directed graph. <todo>
        """
        string = ""
        for traj in self.trajs:
            string += str(traj) + os.linesep
        for node in self.nodes:
            string += str(node) + os.linesep
        string.strip()
        return string