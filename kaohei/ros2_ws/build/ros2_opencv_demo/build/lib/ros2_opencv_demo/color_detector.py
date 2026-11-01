import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.cv_bridge = CvBridge()

        self.target_color = 'red'

        # 每项为 (lower1, upper1, lower2, upper2)：颜色只需一段阈值时后两项填 None
        self.color_ranges = {
            'red': ([0, 43, 46], [10, 255, 255], [156, 43, 46], [180, 255, 255]),
            'green': ([35, 43, 36], [77, 255, 255], None, None),
            'blue': ([100, 43, 46], [124, 255, 255], None, None),
        }

        self.get_logger().info(f'颜色识别节点已启动，正在识别{self.target_color}颜色...')

    def image_callback(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')
            return

        # 图像预处理
        blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
        hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 获取目标颜色和阈值
        lower1, upper1, lower2, upper2 = self.color_ranges[self.target_color]

        # 生成颜色掩码
        mask = cv2.inRange(hsv_image, np.array(lower1), np.array(upper1))
        if lower2 is not None and upper2 is not None:
            mask2 = cv2.inRange(hsv_image, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask, mask2)

        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        # 轮廓检测
        contours, _ = cv2.findContours(
            closing,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 轮廓处理
        found = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 面积过滤
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                # 绘制结果
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(
                    cv_image,
                    f'({center_x},{center_y})',
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                self.get_logger().info(
                    f'检测到{self.target_color}色目标,中心坐标：{center_x},{center_y}'
                )
                found = True

        if not found:
            self.get_logger().debug(f'未检测到{self.target_color}色目标')

        # 显示结果
        cv2.imshow('颜色识别结果', cv_image)
        cv2.imshow('颜色掩码', closing)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    color_detector = ColorDetector()
    try:
        rclpy.spin(color_detector)
    except KeyboardInterrupt:
        pass
    finally:
        color_detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
