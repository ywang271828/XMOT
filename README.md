# XPCI Multi Object Tracker

A library for automatic quantitative characterization of X-ray Phase Contrasting Imaging videos of combustions.

## Installation
We recommend using the library in a Unix/Linux system. For Windows users, the `Windowns Subsystem for Linux (WSL)` can be easily installed following the instructions [here](https://learn.microsoft.com/en-us/windows/wsl/install).

1. Install `git` if you don't already have one. For example, on Ubuntu,
    ```bash
    sudo apt-get install git
    ```

2. Download the `xmot` source code.
    ```bash
    git clone https://github.com/ywang271828/XMOT.git
    cd XMOT
    ```

3. Install Poetry (if not already installed)

    Poetry manages dependencies and virtual environments through the pyproject.toml file.

    * Recommended way (via the official installer):

        ```bash
        curl -sSL https://install.python-poetry.org | python3 -
        ```

        After installation, make sure Poetry is on your PATH (add ~/.local/bin to PATH if needed).

    * Alternative (via pip):

        ```bash
        pip install poetry
        ```

    Check installation:
    ```bash
    poetry --version
    ```

4. Install dependencies and the package

    Poetry will resolve and install dependencies based on the `pyproject.toml` file.

    ```bash
    cd mot
    poetry install
    ```

    By default, `poetry` will use the `poetry.lock` file and install the exact versions listed in
    the `poetry.lock` file. To upgrade dependencies to the latest versions, run `poetry update`.


5. Check whether the installation has succeeded. If the installation succeeded without a problem, users should be able to import `xmot` without errors.

    ```bash
    python

    # In the python prompt
    > import xmot
    ```

## Usage
The workflow for analyzing a video with this package can be roughly divided into three steps:
1.  Detecting particles in each frame of the video
2.  Building trajectories from detections in each frame
2.  Analyzing trajetories to extract statistics and events;

We provide a simple example in the `examples/GMM` folder. Below we explain how to run it and what outputs to expect.

1. Unzip example images.

    The example images are provided in `.zip` format on GitHub for easier version control.
    After unzipping, you will find two datasets:
    * Original XPCI frames
    * Frames pre-processed with the background subtraction procedure described in the paper

    ```bash
    cd XMOT/examples/GMM
    unzip frames_orig.zip
    unzip frames_brightfield_subtracted.zip
    ```

2. Run the demo script
    ```
    python demo.py
    ```

    When finished, an output folder will be created. It contains both the statistics extracted from the digraph representation of the example video and intermediate results for debugging and reference.

3. Understand the output

    The main outputs are:

    * `background`: visual representation of the trained GMM background models
    * `plain_foreground`: raw foreground masks from the GMMs
    * `processed_foreground`: foreground masks after morphological operations to remove noises
    * `contour_as_masks` and `centroid`: detected particles shown with contours and their centroids
    * `GMM_n.png`: original frames with detected contours overlaid
    * `kalman`: trajectory construction results; same trajectories across frames are marked with the
    same bounding-box color
    * `events`: frames around detected event time points
    * `gmm_cnt.npy` and `gmm_bbox.txt`: raw contours and bounding boxes from the GMM detector
    * `blobs.txt`: trajectories at each frame after Kalman filtering and Hungarian assignment
    * `digraph.txt`: detailed digraph output with three sections:
        * trajectory statistics
        * event information
        * frame-by-frame particle details for each trajectory

    Since many of the outputs are intended for debugging and reference, only a subset of frames are
    drawn. You can control this by adjusting the parameter debug_image_interval at the top of
    `demo.py.` A smaller interval will generate more debug images.
