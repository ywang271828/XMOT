import sys
import math
from typing import List, Tuple
import pandas as pd
from os import path, listdir
from PIL import Image
import cv2 as cv
from skimage import exposure
import numpy as np

from xmot.logger import Logger
from xmot.digraph.particle import Particle

BACK_TRACE_LIMIT = 3    # Time frames before the start time of a trajectory allowed to
                        # estimate position at.
CLOSE_IN_TIME = BACK_TRACE_LIMIT
CLOSE_IN_SPACE = 40
#EVENT_TIME_WINDOW = 3   # Span of time frames allowed between trajectory start times
#                        # to be considered as candidates for events.

def distance(a: List[float], b: List[float]) -> float:
    """L2 norm of vectors of any dimension."""
    if a is None or b is None:
        Logger.debug("Trying to compute distance for null vectors.")
        return float("inf")
    if len(a) != len(b):
        Logger.error("Cannot calculate distance between two vectors of different dimensions: " + \
                     "{:d} {:d}".format(len(a), len(b)))
        return float("inf")
    sum = 0
    for i in range(0, len(a)):
        sum += (a[i] - b[i])**2
    return math.sqrt(sum)

def vector_angle(v: np.array, u: np.array) -> float:
    """
    Calculate the angle in degrees between two vectors.
    """
    v_norm = np.linalg.norm(v)
    u_norm = np.linalg.norm(u)
    if v_norm == 0 or u_norm == 0:
        return float("nan")

    cos = np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u))

    # Round because of numerical precision
    if cos > 1 and cos - 1 < 0.0001:
        cos = 1
    if cos < -1 and -1 - cos < 0.0001:
        cos = -1

    return np.degrees(np.arccos(cos))

def ptcl_distance(p1, p2):
    return distance(p1.get_position(), p2.get_position())


def traj_distance(t1, t2) -> float:
    """
        Calculate the nearest distance between two trajectories during the video.

        If t1 ends before (starts late) than t2, calculate the distance between the
        end (start) of t1 and start (end) of t2. If two trajectory are too far away
        in time (difference larger than self.CLOSE_IN_TIME), they shouldn't have
        any relation and distance is set to infinity.

        If t1 and t2 coexist for some time, calculate the nearest distance during these
        frames of the video.
    """
    t1.sort_particles()
    t2.sort_particles()
    if t1.get_end_time() < t2.get_start_time():
        if t2.get_start_time() - t1.get_end_time() <= CLOSE_IN_TIME:
            return distance(t1.get_position_end(), t2.get_position_start())
        else:
            return float("inf")
    elif t1.get_start_time() > t2.get_end_time():
        if t1.get_start_time() - t2.get_end_time() <= CLOSE_IN_TIME:
            return distance(t1.get_position_start(), t2.get_position_end())
        else:
            return float("inf")
    else:
        start_time = max(t1.get_start_time(), t2.get_start_time())
        end_time = min(t1.get_end_time(), t2.get_end_time())
        t1_ptcls = t1.get_snapshots(start_time, end_time) # Deepcopy of particles
        t2_ptcls = t2.get_snapshots(start_time, end_time)
        min_dist = float("inf")
        for p1 in t1_ptcls:
            last_index = -1
            # t1, t2 are sorted. So the particle of t2 that exists at same time as next particle
            # of t1 must have larger index.
            for i in range(last_index + 1, len(t2_ptcls)):
                p2 = t2_ptcls[i]
                if p2.time_frame == p1.time_frame:
                    dist = distance(p1.get_position(), p2.get_position())
                    min_dist = dist if dist < min_dist else min_dist
                    last_index = i
                    break
        return min_dist

def extract_images(video: str, to_gray = False):
    images = []
    video_cap = cv.VideoCapture(video)
    while True:
        ret, img = video_cap.read()
        if not ret: break
        if to_gray:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        images.append(img)
    return images

def collect_images(dir: str, prefix: str, ext: str, start: int, end: int) \
    -> List[Image.Image]:
    files = [f for f in listdir(dir)
              if f.startswith(prefix) and f.endswith(ext)]
    files.sort(key=lambda f: int(f.replace(prefix, "").replace("." + ext, "")))
    numbers = [int(f.replace(prefix, "").replace("." + ext, "")) for f in files]
    if start != -1:
        # There is no guarantee the file names start with number 1, so use remove()
        # instead of del, which requires an index.
        for i in range(numbers[0], start):
            files.remove("{:s}{:d}.{:s}".format(prefix, i, ext))

    if end != sys.maxsize:
        for i in range(end + 1, numbers[-1] + 1): # end is inclusive
            files.remove("{:s}{:d}.{:s}".format(prefix, i, ext))
    images = [Image.open(path.join(dir, f)).copy() for f in files] # Use copy() to retain the image but close the file descriptor.
    return images

def paste_images(left_imgs: List[Image.Image], right_imgs: List[Image.Image], dest, write_img, ids=None) \
    -> List[Image.Image]:

    images = []
    # original image and reproduced image should have same height.
    if len(left_imgs) != len(right_imgs):
        Logger.warning("There aren't same number of detection and reproduced iamges! Reproduced video will only be generated for the frames having detection images.")

    new_res = (left_imgs[0].width + right_imgs[0].width, left_imgs[0].height)
    for i in range(len(left_imgs)):
        im = Image.new("RGBA", new_res)
        im.paste(left_imgs[i], box=(0, 0)) # top left cornor of the box to paste the picture.
        im.paste(right_imgs[i], box=(left_imgs[i].width + 1, 0))
        if write_img:
            if ids != None:
                im.save(path.join(dest, "merged_{:d}.png".format(ids[i])))
            else:
                im.save(path.join(dest, "merged_{:d}.png".format(i)))
        images.append(im)
    return images

def generate_video(images, output: str, fps: int = 24,
                   res: Tuple[int] = None, format: str = "avi"):
    """
        Generate video from given list of iamges.

        Args:
            images: List[np.array] List of images in openCV format (i.e. np.array).
            output: Name of the video. If extension is given, format is ignored.
            fps: Frame rate of the video.
            res: Resolution of the video in pixel. If not given, the largest of
                all images are used to accommodate all images.
            format: Format of the video. Ignored when file extension is given in
                name.
    """
    # Collect info from images
    # If resolution not given, use maximum size of all images.
    if res is None:
        # image are numpy.ndarray. image.shape = (height, width, number of color channels)
        max_height = max([i.shape[0] for i in images])
        max_width = max([i.shape[1] for i in images])
        res = (max_width, max_height)

    if "." not in output:
        output = "{:s}.{:s}".format(output, format)

    fourcc = cv.VideoWriter_fourcc(*'XVID') # XVID: .avi; mp4v: .mp4
    video = cv.VideoWriter(output, fourcc, fps, res) # res is (width, height)
    for i in images:
        video.write(i)
    video.release() # generate video.
    cv.destroyAllWindows()

def contrast_stretch(img, saturation=2.0):
    """
    Simple linear contrast enhancement. See reference: https://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm
    """
    lower_limit, upper_limit = np.percentile(img, (saturation, 100 - saturation))
    img_stretched = exposure.rescale_intensity(img, in_range=(lower_limit, upper_limit))
    return img_stretched

def torch_bbox_to_coordinates(bbox) -> List[Tuple[int]]:
    """
    Transform bbox in torch format to pair of coordinates.
    """
    return [(bbox[0], bbox[1]), (bbox[2], bbox[3])]