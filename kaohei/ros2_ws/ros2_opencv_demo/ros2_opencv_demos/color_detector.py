import rclpy                                     # ROS 2 Python 客户端库
from rclpy.node import Node                      # 节点基类
from sensor_msgs.msg import Image                # ROS 标准图像消息类型
from cv_bridge import CvBridge                   # ROS 图像 <-> OpenCV 图像 转换工具
import cv2                                       # OpenCV：图像处理与显示
import numpy as np                               # 数值计算：构造 HSV 阈值数组、形态学核


class ColorDetector(Node):                                # 颜色识别节点：订阅相机图像，识别指定颜色目标
    def __init__(self):                                   # 构造函数：建立订阅、初始化转换器与阈值表
        super().__init__('color_detector')                # 初始化父类，节点名 color_detector
        self.subscription = self.create_subscription(     # 创建图像订阅者
            Image,                                        # 消息类型：图像
            '/camera/image_raw',                          # 话题名：原始相机图像
            self.image_callback,                          # 回调：每来一帧调用一次
            10                                            # 队列深度：最多缓存 10 条消息
        )

        self.cv_bridge = CvBridge()                       # 图像转换器，实例化一次重复使用

        self.target_color = 'red'                         # 要识别的颜色，可选值见下面的 color_ranges

        # 每项为 (lower1, upper1, lower2, upper2)：颜色只需一段阈值时后两项填 None
        self.color_ranges = {                                                                  # 颜色 -> HSV 阈值表
            'red': ([0, 43, 46], [10, 255, 255], [156, 43, 46], [180, 255, 255]),              # 红色跨 0 度，需两段阈值
            'green': ([35, 43, 36], [77, 255, 255], None, None),                               # 绿色一段阈值即可
            'blue': ([100, 43, 46], [124, 255, 255], None, None),                              # 蓝色一段阈值即可
        }

        self.get_logger().info(f'颜色识别节点已启动，正在识别{self.target_color}颜色...')   # 启动日志

    def image_callback(self, msg):   # 订阅回调：处理每一帧
        try:   # 转换可能抛异常（编码不符/空帧）
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')   # ROS 图像 -> OpenCV BGR 图像
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')   # 转换失败只记日志
            return   # 丢弃这一帧，不中断节点

        # 图像预处理
        blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)                     # 高斯模糊：5x5 核，抑制噪声
        hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)                # BGR -> HSV，便于按颜色分割

        # 获取目标颜色和阈值
        lower1, upper1, lower2, upper2 = self.color_ranges[self.target_color]   # 解包两段阈值（第二段可能为 None）

        # 生成颜色掩码
        mask = cv2.inRange(hsv_image, np.array(lower1), np.array(upper1))   # 第一段范围内的像素置 255，其余 0
        if lower2 is not None and upper2 is not None:                       # 存在第二段阈值（红色才有）
            mask2 = cv2.inRange(hsv_image, np.array(lower2), np.array(upper2))   # 第二段范围掩码
            mask = cv2.bitwise_or(mask, mask2)                                   # 两段掩码按位或合并

        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)                                # 5x5 形态学结构元素
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)          # 开运算：去掉小噪点
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)      # 闭运算：填补目标内部孔洞

        # 轮廓检测
        contours, _ = cv2.findContours(       # 查找轮廓，第二个返回值是层级信息，此处不用
            closing,                          # 输入：形态学处理后的掩码
            cv2.RETR_EXTERNAL,                # 只提取最外层轮廓
            cv2.CHAIN_APPROX_SIMPLE           # 压缩轮廓点，节省内存
        )

        # 轮廓处理
        found = False                                                   # 本帧是否检测到目标的标志
        for contour in contours:                                        # 遍历每个轮廓
            area = cv2.contourArea(contour)                             # 计算轮廓面积（像素数）
            if area > 500:  # 面积过滤，小于 500 像素视为噪点
                x, y, w, h = cv2.boundingRect(contour)                  # 最小外接正矩形：左上角 + 宽高
                center_x = x + w // 2                                   # 矩形中心 x 坐标
                center_y = y + h // 2                                   # 矩形中心 y 坐标

                # 绘制结果
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)   # 画绿色矩形框，线宽 2
                cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)    # 画红色实心圆标出中心点
                cv2.putText(                     # 在框上方写中心坐标
                    cv_image,                    # 目标图像
                    f'({center_x},{center_y})',  # 文本内容
                    (x, y - 10),                 # 文本左下角位置（框上方 10 像素）
                    cv2.FONT_HERSHEY_SIMPLEX,    # 字体
                    0.5,                         # 字号缩放
                    (0, 255, 0),                 # 颜色：绿色
                    2                            # 线宽
                )
                self.get_logger().info(          # 打印检测结果日志
                    f'检测到{self.target_color}色目标,中心坐标：{center_x},{center_y}'
                )
                found = True                     # 标记本帧已检测到目标

        if not found:
            self.get_logger().debug(f'未检测到{self.target_color}色目标')   # 调试日志，默认级别下不输出

        # 显示结果
        cv2.imshow('颜色识别结果', cv_image)     # 窗口：画好框的原图
        cv2.imshow('颜色掩码', closing)          # 窗口：最终掩码
        cv2.waitKey(1)                           # 处理窗口事件 1ms，否则画面不刷新


def main(args=None):
    rclpy.init(args=args)                 # 初始化 ROS 2 上下文
    color_detector = ColorDetector()      # 创建节点
    try:
        rclpy.spin(color_detector)        # 事件循环：等待图像消息并触发回调
    except KeyboardInterrupt:             # 捕获 Ctrl+C
        pass                              # 忽略，交给 finally 统一清理
    finally:
        color_detector.destroy_node()     # 销毁节点，释放订阅等资源
        rclpy.shutdown()                  # 关闭 ROS 2 上下文，与 init 成对
        cv2.destroyAllWindows()           # 关闭所有 OpenCV 窗口


if __name__ == '__main__':     # 仅当直接运行本文件时生效
    main()                     # 进入主函数
