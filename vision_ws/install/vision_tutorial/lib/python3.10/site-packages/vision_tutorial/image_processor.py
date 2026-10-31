#!/usr/bin/env python3  
import rclpy  # 导入 ROS 2 的 Python 客户端库，用于初始化/关闭 ROS 通信、启动自旋等
from rclpy.node import Node  # 导入 Node 类，作为自定义 ROS 2 节点的父类
from sensor_msgs.msg import Image  # 导入标准图像消息类型 Image（sensor_msgs 包中定义）
from cv_bridge import CvBridge  # 导入 CvBridge，用于 ROS 图像消息与 OpenCV 图像之间的相互转换
import cv2  # 导入 OpenCV 库，用于颜色转换、边缘检测、图像显示等操作
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy  # 导入 QoS 配置类及可靠性、持久性策略枚举


class ImageProcessor(Node):  # 定义图像处理节点类，继承 ROS 2 的 Node 基类，从而拥有节点能力
    def __init__(self):  # 构造函数：创建节点实例时自动执行，完成初始化工作
        super().__init__('image_processor')  # 调用父类构造函数，并把节点命名为 'image_processor'
        qos_profile = QoSProfile(  # 创建 QoS（服务质量）配置对象，用于规定通信的可靠性策略
            depth=5,  # 队列深度为 5：最多缓存 5 条未处理的消息，超出则丢弃最旧的
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # 可靠性为“尽力而为”：不保证送达，减少延迟，常用于相机高频数据
            durability=QoSDurabilityPolicy.VOLATILE  # 持久性为“易失”：订阅者只能收到订阅之后发布的消息
        )  # QoS 配置对象创建结束


        self.subscription = self.create_subscription(  # 创建订阅者，并保存到成员变量防止被垃圾回收导致订阅失效
            Image,  # 订阅的消息类型：sensor_msgs/msg/Image
            '/camera/image_raw',  # 订阅的话题名：相机原始图像话题
            self.image_callback,  # 每收到一条消息就调用该回调函数处理
            qos_profile  # 使用上面定义的 QoS 配置
        )  # 订阅者创建结束
        self.bridge = CvBridge()  # 实例化 CvBridge，后续用于把 ROS 消息转换为 OpenCV 图像
        self.get_logger().info('图像处理节点已启动，正在等待图像数据...')  # 输出一条 INFO 级日志，提示节点已启动
        
        
        
    def image_callback(self, msg):  # 回调函数：收到一帧图像消息 msg 时被调用
        try:  # 开始异常保护，避免转换失败导致节点崩溃
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # //将ROS图像消息转换为OpenCV图像，BGR格式
        except Exception as e:  # 捕获转换过程中出现的任何异常
            self.get_logger().error(f'图像CvBridge转换失败: {e}')  # 输出 ERROR 级日志，包含具体异常信息
            return  # 转换失败则直接返回，跳过本帧后续处理
        try:  # 开始第二段异常保护，防止 OpenCV 处理出错
            grag = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)  # 彩色图转灰度图，便于做边缘检测（变量名可能是 gray 的笔误）
            edges = cv2.Canny(grag, 50, 150)  # 用 Canny 算法做边缘检测，低阈值 50、高阈值 150
            cv2.imshow('原始图像',cv_image)  # 弹出窗口显示原始彩色图像
            cv2.imshow('边缘检测图像', edges)  # 弹出窗口显示边缘检测结果（黑白边缘图）
            cv2.waitKey(1)  # 等待 1 毫秒并刷新窗口，让图像能够真正显示出来（必须有它窗口才会刷新）
        except Exception as e:  # 捕获 OpenCV 处理过程中的任何异常
            self.get_logger().error(f'OpenCV处理失败: {e}')  # 输出 ERROR 级日志，记录处理失败原因




def main(args=None):  # 程序主入口函数，args 用于接收命令行参数（如 ROS remap 参数）
    rclpy.init(args=args)  # 初始化 ROS 2 通信系统（必须在使用任何 ROS 功能之前调用）
    node = ImageProcessor()  # 创建并启动图像处理节点实例
    try:  # 进入自旋，捕获人工停止（Ctrl+C）的情况
        rclpy.spin(node)  # 让节点进入事件循环，持续接收并处理回调（此处会一直阻塞直到退出）
    except KeyboardInterrupt:  # 用户按下 Ctrl+C 时抛出此异常，在这里优雅捕获
        node.get_logger().info('图像处理节点已手动关闭')  # 打印一条日志，告知节点被手动停止
    finally:  # 无论正常退出还是异常退出，都执行清理工作
        cv2.destroyAllWindows()  # 关闭所有由 OpenCV 创建的显示窗口，释放 GUI 资源
        node.destroy_node()  # 销毁节点，释放其占用的订阅、发布等资源
        rclpy.shutdown()  # 关闭 ROS 2 通信系统，释放上下文资源



if __name__ == '__main__':  # 判断当前脚本是否被直接运行（而不是被 import 导入）
    main()  # 直接运行时调用主函数启动程序