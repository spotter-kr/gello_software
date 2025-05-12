from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from gello.agents.gello_agent import GelloAgent

import rclpy
from rclpy.node import Node
from gello_msgs.msg import GelloState

@dataclass
class Args:
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hz: int = 50
    start_joints: Tuple[float, ...] = (0,0,np.pi/2,0,np.pi/2,0,0)

    gello_port: str = '/dev/ttyUSB0'
    verbose: bool = False
    
class GelloNode(Node):
    def __init__(self, args: Args):
        super().__init__('gello_node')

        gello_port = args.gello_port
        self._agent = GelloAgent(
            port=gello_port,
            type='tm', 
            start_joints=np.array(args.start_joints)
        )
    
        self._tm_publisher = self.create_publisher(GelloState, '/gello_state', 10)
        timer_period = 1 / args.hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        positions = self._agent.act({})
        msg = GelloState()
        msg.joints = np.rad2deg(positions[:6]).tolist()
        msg.gripper = float(positions[6])
        self._tm_publisher.publish(msg)

def main():
    rclpy.init()
    gello_node = GelloNode(Args())
    rclpy.spin(gello_node)
    gello_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

