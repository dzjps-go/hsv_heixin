import rclpy                      # ROS 2 的 Python 客户端库，提供 init/spin/shutdown 等接口
from rclpy.node import Node       # 节点基类，自定义节点都需要继承它
from sensor_msgs.msg import Image  # ROS 标准图像消息类型 sensor_msgs/Image
from cv_bridge import CvBridge    # ROS 图像消息 <-> OpenCV/NumPy 图像的转换工具
import cv2                        # OpenCV，用于图像显示与处理


class ImageSubscriber(Node):          # 定义节点类，继承 rclpy 的 Node
    def __init__(self):
        # 初始化父类，'image_subscriber' 是节点在 ROS 图中的名字（ros2 node list 可见）
        super().__init__('image_subscriber')

        # 创建订阅者，订阅摄像头图像话题，每收到一帧就触发 image_callback
        self.subscription = self.create_subscription(
            Image,                  # 消息类型：sensor_msgs/Image
            '/camera/image_raw',    # 话题名，必须与相机节点发布的话题一致
            self.image_callback,    # 收到消息后调用的回调函数
            10)                     # QoS 队列深度：最多缓存 10 条待处理消息，超出丢弃最旧的
        # 读一次属性，避免静态检查工具提示“变量未使用”；真正让订阅保持存活的是上面赋给 self.subscription
        self.subscription



        # 创建 CvBridge 对象并复用（不要每次回调都新建），用于将 ROS 图像消息转换为 OpenCV 图像
        self.cv_bridge = CvBridge()
        # 打印启动日志，方便确认节点已启动、正在等待图像数据
        self.get_logger().info('图像订阅节点已启动，正在等待图像数据...')


    def image_callback(self, msg):
        # 订阅回调：msg 是一条 sensor_msgs/Image 消息，每当收到新图像时被调用
        try:
            # 将 ROS 图像消息转换为 OpenCV 图像（NumPy 数组）
            # desired_encoding='bgr8' 表示无论原始编码如何，都统一转成 BGR 三通道 8 位格式
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            # 转换失败（编码不支持、数据为空等）时记录错误，并丢弃这一帧
            self.get_logger().error(f'图像CvBridge转换失败: {e}')
            return



        # 显示原始图像：'摄像头图像' 是窗口名，同名窗口会被复用而不会重复弹出
        cv2.imshow('摄像头图像', cv_image)
        # 必须调用：给 OpenCV 1 毫秒处理窗口事件（重绘、按键），否则窗口无法显示
        cv2.waitKey(1)

        # 图像预处理：先做高斯模糊去噪，减少后续按颜色分割时的噪点
        # (5, 5) 是卷积核大小（必须是正奇数），sigmaX=0 表示由核大小自动推算标准差
        blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)

        # 转成 HSV 色彩空间：H 0~179、S/V 0~255，比 BGR 更适合按颜色分割（如黑色检测）
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        



def main(args=None):
    # 初始化 ROS 2 通信上下文，必须在创建节点之前调用
    rclpy.init(args=args)
    # 实例化节点，构造时已完成订阅者注册
    image_subscriber = ImageSubscriber()
    try:
        # 进入事件循环并阻塞，持续把收到的消息分发给 image_callback
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        # 捕获 Ctrl+C，正常退出而不打印异常栈
        pass
    finally:
        # 无论正常退出还是异常，都执行清理
        image_subscriber.destroy_node()  # 销毁节点，释放订阅、日志等资源
        rclpy.shutdown()                 # 关闭 ROS 2 上下文，与 rclpy.init 成对
        cv2.destroyAllWindows()          # 关闭所有 OpenCV 窗口，释放 GUI 资源



if __name__ == '__main__':  # 仅当直接运行本文件时才执行（被 import 时不会启动）
    main()
