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

    By default, `poetry` will use the resolved lock file to install the exact versions listed in
    the `poetry.lock` file. To upgrade them to the latest versions, run `poetry update`.


5. Check whether the installation has succeeded. If the installation succeeded without a problem, users should be able to import `xmot` without errors.

    ```bash
    python

    # In the python prompt
    > import xmot
    ```

## Usage
The process of analyzing a video can be largely separated into two steps:
1.  Detecting particles from each frame of the video;
2.  Analyzing detected particles and extract information;

We have included example scripts for the two steps in the folder `examples/video_1`:
`particle_detection.py` and `particle_analysis.py`. The first script will extract frames from the
vedio and detect particles from each of the frame. The second script parses the output from the
first script and try to build the graph representation fo the video, from which we can detect
events. The second script will write out particle and trajectory information to the stdout. To save them into a file, use redirection `python particle_analysis.py > output.txt`.

In both scripts, the input parameters that users might need to adjust are listed at the beginning. Out of them, the following variables could use additional explanation:

`particle_detection.py`:

- `model`:
    * Permitted values:
        * Path to the pre-trained model
        * `None`
    * Note:
        * Path to the pre-trained model, if existing.
        * If users want to train a new model based on the input video, use "None".

- `commons.PIC_DIMENSION`:
    * Permitted values:
        * List of two integers specifying the resolution of the video. E.g. `[624, 640]` for videos with 90kfps.
    * Note:
        * A smaller image of this size will be cropped out from each video frame for detection and analysis. If users want to use
        the full image, remove this variable and the resolution will
        be read from the video.
        * In some XPCI videos, there are time stamps on the upper right cornor which would be mistakenly recoganized as particles. To remove them, we can define this parameter.
        * This parameter is also used during particle detection to filter out invalid bounding boxes of particles which might have coordinates outside the image.
        * If this variable is set in `particle_detection.py`, users need to keep the same variable consistent in `particle_analysis.py`.

- `device`:
    * Permitted values: `cuda`, `cuda:0`, `cpu`
    * Note:
        * This variable is passed onto the function `xmot.mot.identifier::identify()` to set the device to used for training and running of the machine-learning model.
        * To check whether CUDA is available, run
            ```
            python
            > import torch
            > torch.cuda.is_available()
            ```
            If CUDA is available, the output should be `True`. Otherwise, it will be `False`.
