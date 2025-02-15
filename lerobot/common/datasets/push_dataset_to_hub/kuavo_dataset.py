#!/usr/bin/env python3
# pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag
# pip install roslz4 --extra-index-url https://rospypi.github.io/simple/
import numpy as np
import cv2
import rosbag
from pprint import pprint


# ================ 机器人关节信息定义 ================

DEFAULT_LEG_JOINT_NAMES=[
    "l_leg_roll", "l_leg_yaw", "l_leg_pitch", "l_knee", "l_foot_pitch", "l_foot_roll",
    "r_leg_roll", "r_leg_yaw", "r_leg_pitch", "r_knee", "r_foot_pitch", "r_foot_roll",
]
DEFAULT_ARM_JOINT_NAMES = [
    "zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link", "zarm_l5_link", "zarm_l6_link", "zarm_l7_link",
    "zarm_r1_link", "zarm_r2_link", "zarm_r3_link", "zarm_r4_link", "zarm_r5_link", "zarm_r6_link", "zarm_r7_link",
]
DEFAULT_HEAD_JOINT_NAMES = [
    "head_yaw", "head_pitch"
]

DEFAULT_JOINT_NAMES = {
    "full_joint_names": DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + DEFAULT_HEAD_JOINT_NAMES,
    "leg_joint_names": DEFAULT_LEG_JOINT_NAMES,
    "arm_joint_names": DEFAULT_ARM_JOINT_NAMES,
    "head_joint_names": DEFAULT_HEAD_JOINT_NAMES,
}

# ================ 机器人话题信息定义 ================
DEFAULT_CAMERA_NAMES = ["camera", "cam_high", "cam_low", "cam_right_wrist", "cam_left_wrist"]

# ================ 数据处理函数定义 ==================

class KuavoMsgProcesser:
    """
    Kuavo 话题处理函数
    """
    @staticmethod
    def process_color_image(msg):
        """
        Process the color image.
        Args:
            msg (sensor_msgs.msg.Image): The color image message.
        Returns:
             Dict:
                - data(np.ndarray): Image data with shape (height, width, 3).
                - "timestamp" (float): The timestamp of the image.
        """
        if msg.encoding != 'rgb8':
            # Handle different encodings here if necessary
            raise ValueError(f"Unsupported encoding: {msg.encoding}. Expected 'rgb8'.")

        # Convert the ROS Image message to a numpy array
        img_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

        # If the image is in 'bgr8' format, convert it to 'rgb8'
        if msg.encoding == 'bgr8':
            cv_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        else:
            cv_img = img_arr

        return {"data": cv_img, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_state(msg):
        """
            Args:
                msg (kuavo_msgs/sensorsData): The joint state message.
            Returns:
                Dict:
                    - data(np.ndarray): The joint state data with shape (28,).
                    - "timestamp" (float): The timestamp of the joint state.
        """
        # radian
        joint_q = msg.joint_data.joint_q
        return {"data": joint_q, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_cmd(msg):
        """
            Args:
                msg (kuavo_msgs/jointCmd): The joint state message.

            Returns:
                Dict:
                    - data(np.ndarray): The joint state data with shape (28,).
                    - "timestamp" (float): The timestamp of the joint state.
        """
        # radian
        return {"data": msg.joint_q, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_sensors_data_raw_extract_imu(msg):
        imu_data = msg.imu_data
        gyro = imu_data.gyro
        acc = imu_data.acc
        free_acc = imu_data.free_acc
        quat = imu_data.quat

        # 将数据合并为一个NumPy数组
        imu = np.array([gyro.x, gyro.y, gyro.z,
                        acc.x, acc.y, acc.z,
                        free_acc.x, free_acc.y, free_acc.z,
                        quat.x, quat.y, quat.z, quat.w])

        return {"data": imu, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_sensors_data_raw_extract_arm(msg):
        res = KuavoMsgProcesser.process_joint_state(msg)
        res["data"] = res["data"][12:26]
        return res

    @staticmethod
    def process_joint_cmd_extract_arm(msg):
        res = KuavoMsgProcesser.process_joint_cmd(msg)
        res["data"] = res["data"][12:26]
        return res

    @staticmethod
    def process_sensors_data_raw_extract_arm_head(msg):
        res = KuavoMsgProcesser.process_joint_state(msg)
        res["data"] = res["data"][12:]
        return res

    @staticmethod
    def process_joint_cmd_extract_arm_head(msg):
        res = KuavoMsgProcesser.process_joint_cmd(msg)
        res["data"] = res["data"][12:]
        return res

    @staticmethod
    def process_depth_image(msg):
        """
        Process the depth image.

        Args:
            msg (sensor_msgs/Image): The depth image message.

        Returns:
            Dict:
                - data(np.ndarray): Depth image data with shape (height, width).
                - "timestamp" (float): The timestamp of the image.
        """
        # Check if the image encoding is '16UC1' which is a common encoding for depth images
        if msg.encoding != '16UC1':
            raise ValueError(f"Unsupported encoding: {msg.encoding}. Expected '16UC1'.")

        # Convert the ROS Image message to a numpy array
        img_arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        # The depth image is already in '16UC1' format, so no conversion is needed
        return {"data": img_arr, "timestamp": msg.header.stamp.to_sec()}

class KuavoRosbagReader:
    def __init__(self):
        self._msg_processer = KuavoMsgProcesser()
        self._topic_process_map = {
            "observation.state": {
                "topic": "/sensors_data_raw",
                "msg_process_fn": self._msg_processer.process_joint_state,
            },
            "action": {
                "topic": "/joint_cmd",
                "msg_process_fn": self._msg_processer.process_joint_cmd,
            },
            "observation.imu": {
                "topic": "/sensors_data_raw",
                "msg_process_fn": self._msg_processer.process_sensors_data_raw_extract_imu,
            }
        }
        for camera in DEFAULT_CAMERA_NAMES:
            # observation.images.{camera}.depth  => color images
            self._topic_process_map[f"observation.{camera}"] = {
                "topic": f"/{camera}/color/image_raw",
                "msg_process_fn": self._msg_processer.process_color_image,
            }
            # observation.images.{camera}.depth => depth images
            self._topic_process_map[f"observation.images.{camera}.depth"] = {
                "topic": f"/{camera}/depth/image_rect_raw",
                "msg_process_fn": self._msg_processer.process_depth_image,
            }

    def load_raw_rosbag(self, bag_file: str):
        bag = rosbag.Bag(bag_file)      
        self.print_bag_info(bag)  
        return bag
    
    def print_bag_info(self, bag: rosbag.Bag):
        pprint(bag.get_type_and_topic_info().topics)
    
    def process_rosbag(self, bag_file: str):
        """
        Process the rosbag file and return the processed data.

        Args:
            bag_file (str): The path to the rosbag file.

        Returns:
            Dict: The processed data.
        """
        bag = self.load_raw_rosbag(bag_file)
        data = {}
        for key, topic_info in self._topic_process_map.items():
            topic = topic_info["topic"]
            msg_process_fn = topic_info["msg_process_fn"]
            data[key] = []
            for _, msg, t in bag.read_messages(topics=topic):
                data[key].append(msg_process_fn(msg))
        return data
    
    

if __name__ == '__main__':
    bag_file = '/Users/wason/Code/RobotEmbodiedData/lerobot/data/testcamera/00001/testcamera_20250213_193331.bag'
    reader = KuavoRosbagReader()
    data = reader.process_rosbag(bag_file)
    print(data.keys())

