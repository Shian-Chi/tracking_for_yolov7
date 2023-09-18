import sys
sys.path.append('./yolo')
from yolo.my_detect3 import YOLO, argument, get_detect
from yolo.myProject.dataset import CameraReader, gstreamer_pipeline
import cv2 
import rclpy
from rclpy.node import Node
import time
from geometry_msgs.msg import Twist



class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.Twist_publisher_ = self.create_publisher(Twist, 'topic', 10)

        self.tracker = None
        timer_period = 1/30  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Twist()
        if self.tracker != None:
            fps, flag, yaw, pitch = get_detect(self.tracker)
            print(f'{fps}, {flag}, {yaw}, {pitch}')
            msg.angular.y = pitch
            msg.angular.z = yaw
            self.Twist_publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    '''YOLO'''
    opt = argument(w_target = 'tennisv7.pt', target = 'tennis-ball')
    test = YOLO(opt)
    
    minimal_publisher = MinimalPublisher()
    minimal_publisher.tracker = test

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()