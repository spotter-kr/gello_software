
from typing import Dict, List
import json
import numpy as np
from enum import Enum
from threading import Thread
from queue import Queue
from time import sleep
from enum import Enum
from dataclasses import dataclass
from copy import deepcopy

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import String

from tm_msgs.srv import SendScript, SetIO, SetEvent
from tm_msgs.msg import FeedbackState
from gello_msgs.msg import GelloState
from gello_tm.tm_command import *

DO_STATE_ON = 1.0
DO_STATE_OFF = 0.0

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = 0.0
GRIPPER_NONE = -1.0

START_JOINTS = [0.0,0.0,90.0,0.0,90.0,0.0,0.0]

PROTECTION_20MS = 20
PROTECTION_50MS = 50
DEG_PER_MS = 0.05
    
class TMRobot(Node):
    """The ROS node that sends TMscript to the TM robot in response to action commands.
    Since some commands have to be executed in a certain sequence - while they are naturally not synchronized -
    , this node maintains a command queue to ensure actions are delievered in a right order and timing.

    Subscribes:
        /tm_command (std_msgs.msg.String): The action command to be executed.

    Client:
        /send_script (tm_msgs.srv.SendScript): The TM script service.
    """

    def __init__(self, gripper: bool = False):
        Node.__init__(self, 'tm_node')

        self._command_queue = Queue()
        self._command_thread = Thread(target=self._execute_command, daemon=True)
        self._command_thread.start()
    
        self._gripper = gripper
        self._gripper_state = GRIPPER_NONE
        self._gripper_goal = GRIPPER_NONE
        self._joint_pos = None # rad
        self._joint_vel = None
        self._continuous_mode = False

        tm_cmd_cb_group = MutuallyExclusiveCallbackGroup()
        tm_state_cb_group = MutuallyExclusiveCallbackGroup()

        # TM robotics service clients belong to the dedicated callback group for requests
        self.client_script = self.create_client(SendScript, 'send_script', callback_group=tm_cmd_cb_group)
        while not self.client_script.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('send_script service not available, waiting again...')

        self.client_io = self.create_client(SetIO, 'set_io', callback_group=tm_cmd_cb_group)
        while not self.client_io.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_io service not available, waiting again...')

        self.client_event = self.create_client(SetEvent, 'set_event', callback_group=tm_cmd_cb_group)
        while not self.client_event.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_event service not available, waiting again...')

        # self.create_subscription(String, '/tm_command', self._tm_command_callback, 10)
        self.create_subscription(GelloState, '/gello_state', self._gello_state_callback, 10)
        
        # Subscribe feedback topic from the TM driver
        self.create_subscription(
            msg_type=FeedbackState, 
            topic='/feedback_states', 
            callback=self._feedback_states_callback, 
            qos_profile=1, 
            callback_group=tm_state_cb_group
        )

        self.futures = []
        self.get_logger().info('Executor Node Started. Waiting for action commands ...')

    def setup_robot(self):
        self._enqueue_command(Motion(
            command_type = CommandType.MOTION,
            motion_type=MotionType.PTP,
            motion_format=MotionFormat.J,
            target=START_JOINTS[:6]
        ))
        self.set_continuous_mode(True)

    def reset_robot(self):
        self.set_continuous_mode(False)
        self._enqueue_command(Motion(
            command_type = CommandType.MOTION,
            motion_type=MotionType.PTP,
            motion_format=MotionFormat.J,
            target=START_JOINTS[:6]
        ))

    def spin(self):
        while rclpy.ok():
            rclpy.spin_once(self)
            incomplete_futures = []
            for f in self.futures:
                if f.done():
                    res = f.result()
                    print("Received service result: {}".format(res))
                else:
                    incomplete_futures.append(f)
            self.futures = incomplete_futures
        
    # def _tm_command_callback(self, msg):
    #     self.get_logger().info(f'Received action: {msg.data}')

    #     # Commands sent by the operator
    #     if msg.data == 'open':
    #         self._open_gripper()
    #         return
    #     if msg.data == 'close':
    #         self._close_gripper()
    #         return
        
    #     # Commands sent by task planner
    #     action_dict = json.loads(msg.data
    #                              .replace("'", "\"")
    #                              .replace("True", "true")
    #                              .replace("False", "false"))

    #     action_type = action_dict['action']

    #     command = None
    #     if action_type in ['ptp', 'line']:
    #         command = build_motion_tmscript(action_dict)
    #     elif action_type == 'script':
    #         tmscript = action_dict.get('script', '')
    #     elif action_type == 'position':
    #         tmscript = build_position_tmscript(action_dict)
    #     else:
    #         self.get_logger().info(f'Unknown action type: {action_type}')
        
    #     self._enqueue_command(command)

    def _gello_state_callback(self, msg):
        if not self.continuous_mode_enabled():
            return
        if len(msg.joints) < 6:
            return
        self._enqueue_command(Position(
            command_type=CommandType.POS,
            target=msg.joints,
        ))
        self._enqueue_command(Gripper(
            command_type=CommandType.GRIPPER,
            target=msg.gripper,
        ))

    ### Robot states ###
    def _feedback_states_callback(self, msg):
        self._update_robot_state(msg)
        self._update_ee_state(msg)

    def _update_robot_state(self, msg):
        self._joint_pos = deepcopy(msg.joint_pos)
        self._joint_vel = deepcopy(msg.joint_vel)
    
    def _update_ee_state(self, msg):
        ee_dio = msg.ee_digital_input
        if ee_dio[0] == 0 and ee_dio[1] == 1:
            self._gripper_state = GRIPPER_OPEN
            if self._gripper_goal == GRIPPER_OPEN:
                self._gripper_goal = GRIPPER_NONE
        elif ee_dio[0] == 1 and ee_dio[1] == 0:
            self._gripper_state = GRIPPER_CLOSE
            if self._gripper_goal == GRIPPER_CLOSE:
                self._gripper_goal = GRIPPER_NONE
        elif ee_dio[0] == 0 and ee_dio[1] == 0:
            # Gripper in middle position
            self._gripper_state = GRIPPER_CLOSE
            if self._gripper_goal == GRIPPER_CLOSE:
                self._gripper_goal = GRIPPER_NONE
        else:
            self._gripper_state = GRIPPER_NONE
    
    def _enqueue_command(self, command):
        if command:
            self._command_queue.put(command)
    
    def _execute_command(self):
        """
        Sends commands in the commandQueue in order.
        """
        while True:
            command = self._command_queue.get(block=True, timeout=None)
            
            if command.command_type == CommandType.MOTION:
                req = SendScript.Request()
                req.id = "demo"
                req.script = command.toscript()
                self.client_script.call(req)
            elif command.command_type == CommandType.POS:
                if command.enable is None:
                    command = self._pos_delta_cap(command, PROTECTION_20MS)
                req = SendScript.Request()
                req.id = "demo"
                req.script = command.toscript()
                self.client_script.call(req)
                # self.get_logger().info(command.toscript())    
            elif command.command_type == CommandType.GRIPPER:
                if command.target < 0 or command.target > 1:
                    continue
                if command.target > 0.6:
                    if self._gripper_state == GRIPPER_CLOSE:
                        continue        
                    if self._gripper_goal == GRIPPER_CLOSE:
                        continue
                    self._close_gripper()

                if command.target < 0.4 :
                    if self._gripper_state == GRIPPER_OPEN:
                        continue        
                    if self._gripper_goal == GRIPPER_OPEN:
                        continue
                    self._open_gripper()
            else:
                # Unreachable
                continue

    ### TM srv call ###
    def _close_gripper(self):
        self._gripper_goal = GRIPPER_CLOSE
        req0 = SetIO.Request()
        req0.module = SetIO.Request.MODULE_ENDEFFECTOR
        req0.type = SetIO.Request.TYPE_DIGITAL_OUT
        req0.pin = 0
        req0.state = DO_STATE_OFF
        self.client_io.call(req0)

        req1 = SetIO.Request()
        req1.module = SetIO.Request.MODULE_ENDEFFECTOR
        req1.type = SetIO.Request.TYPE_DIGITAL_OUT
        req1.pin = 1
        req1.state = DO_STATE_ON
        self.client_io.call(req1)

    def _open_gripper(self):
        req0 = SetIO.Request()
        req0.module = SetIO.Request.MODULE_ENDEFFECTOR
        req0.type = SetIO.Request.TYPE_DIGITAL_OUT
        req0.pin = 0
        req0.state = DO_STATE_ON
        self.client_io.call(req0)

        req1 = SetIO.Request()
        req1.module = SetIO.Request.MODULE_ENDEFFECTOR
        req1.type = SetIO.Request.TYPE_DIGITAL_OUT
        req1.pin = 1
        req1.state = DO_STATE_OFF
        self.client_io.call(req1)

    def _send_script_request(self, script):
        if script is None or script == '':
            return
        req = SendScript.Request()
        req.id = "demo"
        req.script = script
        # self.futures.append(self.client_script.call_async(req))
        self.client_script.call(req)

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._gripper:
            return 7
        return 6

    def get_gripper_pos(self) -> float:
        return self._gripper_state

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        if self._joint_pos is None:
            return np.array([])
        robot_joints = self._joint_pos
        if self._gripper:
            gripper_pos = self.get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """

        robot_joints: List = np.rad2deg(joint_state[:6]).tolist()
        self._enqueue_command(Position(
            command_type=CommandType.POS,
            target=robot_joints,
        ))
        if self._gripper:
            gripper_pos = joint_state[-1]
            self._enqueue_command(Gripper(
                command_type=CommandType.GRIPPER,
                target=gripper_pos,
            ))

    def continuous_mode_enabled(self) -> bool:
        """Check if the robot is in continuous control mode.

        Returns:
            bool: True if the robot is in continuous control mode, False otherwise.
        """
        return self._continuous_mode

    def set_continuous_mode(self, enable: bool) -> None:
        """Set the continuous control mode of the robot.

        Args:
            enable (bool): True to enable continuous control mode, False to disable it.
        """
        if enable and not self._continuous_mode:
            print('Enable continuous control')
            self._continuous_mode = True
            self._enqueue_command(Position(
                command_type=CommandType.POS,
                enable=True,
                protection=PROTECTION_20MS,
            ))
        elif not enable and self._continuous_mode:
            self._continuous_mode = False
            self._enqueue_command(Position(
                command_type=CommandType.POS,
                enable=False,
            ))

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]])
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }

def main():
    rclpy.init()
    tm_node = TMRobot(gripper=False)
    tm_node.setup_robot()

    executor = MultiThreadedExecutor()
    executor.add_node(tm_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    
    executor.shutdown()
    tm_node.reset_robot()
    tm_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

