#!/usr/bin/env python

"""
Contains utilities to process raw data format from kuavo rosbag
"""
import re
import warnings
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, Features, Image, Sequence, Value
from typing import Dict

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import calculate_episode_data_index
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame



def check_format(raw_dir: Path) -> bool:
    """Check if the raw data format is correct for the kuavo rosbag format
    """
    assert raw_dir.exists()

    leader_file = list(raw_dir.rglob("*.bag"))
    if len(leader_file) == 0:
        raise ValueError(f"Missing bag files in '{raw_dir}'")
    return True

def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    keypoints_instead_of_image: bool = False,
    encoding: dict | None = None,
):
    pass

def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    cfg: Dict,
    video: bool = True,
    fps: int | None = None,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    pass



if __name__ == '__main__':
    check_format(Path('/Users/wason/Code/RobotEmbodiedData/lerobot/data/testcamera/'))