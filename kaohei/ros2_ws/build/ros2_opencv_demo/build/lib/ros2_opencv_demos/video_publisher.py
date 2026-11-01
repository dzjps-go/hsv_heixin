import os

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class VideoPublisher(Node):
    """把 mp4 文件当作虚拟摄像头，按原帧率发布到 /camera/image_raw。"""

    def __init__(self):
        super().__init__('video_publisher')

        # 默认视频路径：~/OpenCv/hsv_heixin/kaohei/kaoheiku_ws/shopi.mp4
        default_path = os.path.join(
            os.path.expanduser('~'), 'OpenCv', 'hsv_heixin', 'kaohei',
            'kaoheiku_ws', 'shopi.mp4')
        self.declare_parameter('video_path', default_path)
        self.declare_parameter('loop', True)

        self.video_path = self.get_parameter('video_path').value
        self.loop = self.get_parameter('loop').value

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'视频打开失败: {self.video_path}')
            raise SystemExit

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)

        self.timer = self.create_timer(1.0 / max(self.fps, 1.0), self.publish_frame)
        self.get_logger().info(
            f'虚拟摄像头已启动 | {self.video_path} | FPS={self.fps:.2f} -> /camera/image_raw')

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            if not self.loop:
                self.get_logger().info('视频播放完毕')
                self.timer.cancel()
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error('视频回绕失败')
                self.timer.cancel()
                return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self.image_pub.publish(msg)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
