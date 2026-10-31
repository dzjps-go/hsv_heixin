import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        #创建订阅者，订阅摄像头图像话题
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.subscription #防止未使用变量警告



        #创建cv_bridge对象，用于将ROS图像消息转换为OpenCV图像
        self.cv_bridge = CvBridge()
        self.get_logger().info('图像订阅节点已启动，正在等待图像数据...')


    def image_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像CvBridge转换失败: {e}')
            return



        #显示图像
        cv2.imshow('摄像头图像', cv_image)
        cv2.waitKey(1)  # 等待1毫秒刷新窗口(必须写，不然窗口无法显示)



def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        image_subscriber.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()  #关闭所有OpenCV窗口，释放资源



if __name__ == '__main__':
    main()
