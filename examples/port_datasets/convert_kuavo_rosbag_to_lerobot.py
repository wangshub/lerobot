"""
Script to convert Kuavo rosbag data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm

from lerobot.common.datasets.push_dataset_to_hub.kuavo_dataset import (
    KuavoRosbagReader,
    DEFAULT_JOINT_NAMES_LIST,
    DEFAULT_CAMERA_NAMES
    )


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None

DEFAULT_DATASET_CONFIG = DatasetConfig()


def get_cameras(bag_data: dict) -> list[str]:
    """
    /camera/color/camera_info           : sensor_msgs/CameraInfo
    /camera/color/image_raw             : sensor_msgs/Image     
    /camera/depth/camera_info           : sensor_msgs/CameraInfo
    /camera/depth/image_rect_raw        : sensor_msgs/Image     
    """
    cameras = []
    for k in bag_data.keys():
        if 'camera' in k and len(bag_data[k]) > 0:
            cameras.append(k)
    return cameras


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    
    motors = DEFAULT_JOINT_NAMES_LIST
    # TODO: auto detect cameras
    cameras = [
        'camera',
        # 'camera.depth',
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        if 'depth' in cam:
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (480, 640),
                "names": [
                    "height",
                    "width",
                ],
            }
        else:
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (3, 480, 640),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def load_raw_images_per_camera(bag_data: dict) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in get_cameras(bag_data):
        imgs_per_cam[camera] = np.array([msg['data'] for msg in bag_data[camera]])
    
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    
    bag_reader = KuavoRosbagReader()
    bag_data = bag_reader.process_rosbag(ep_path)
    
    state = np.array([msg['data'] for msg in bag_data['observation.state']], dtype=np.float32)
    action = np.array([msg['data'] for msg in bag_data['action']], dtype=np.float32)
    
    velocity = None
    effort = None
    
    imgs_per_cam = load_raw_images_per_camera(bag_data)
    
    return imgs_per_cam, state, action, velocity, effort


def diagnose_frame_data(data):
    for k, v in data.items():
        print(f"Field: {k}")
        print(f"  Shape    : {v.shape}")
        print(f"  Dtype    : {v.dtype}")
        print(f"  Type     : {type(v).__name__}")
        print("-" * 40)


def populate_dataset(
    dataset: LeRobotDataset,
    bag_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(bag_files))
    
    for ep_idx in tqdm.tqdm(episodes):
        ep_path = bag_files[ep_idx]
        
        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]
        
        for i in range(num_frames):
            frame = {
                "observation.state": torch.from_numpy(state[i]).type(torch.float32),
                "action": torch.from_numpy(action[i]).type(torch.float32),
            }
            
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]
            
            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]   
            
            # diagnose_frame_data(frame)
            dataset.add_frame(frame)
            
        dataset.save_episode(task=task)

    return dataset
            


def port_kuavo_rosbag(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    # Download raw data if not exists
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)
        
    bag_reader = KuavoRosbagReader() 
    bag_files = bag_reader.list_bag_files(raw_dir)
    
    dataset = create_empty_dataset(
        repo_id,
        robot_type="kuavo4pro",
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        bag_files,
        task=task,
        episodes=episodes,
    )
    dataset.consolidate()
    


if __name__ == "__main__":
    raw_dir = Path('/Users/wason/Code/RobotEmbodiedData/lerobot/data/testcamera2/')
    repo_id = 'lejurobot/kuavo_demo'
    port_kuavo_rosbag(raw_dir, repo_id)