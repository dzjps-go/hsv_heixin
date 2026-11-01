import os                                            # 标准库：用于设置进程环境变量
os.environ["PYTORCH_JIT"] = "0"                      # 关闭 PyTorch JIT，避免与 ROS 2 环境下的算子编译冲突
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "local"     # 让 setuptools 使用自带的 distutils，规避 Python 3.12+ 的兼容问题
import rclpy                                         # ROS 2 的 Python 客户端库
from rclpy.node import Node                          # 节点基类
from sensor_msgs.msg import Image                    # ROS 标准图像消息类型
from cv_bridge import CvBridge                       # ROS 图像消息 <-> OpenCV 图像 的转换工具
from ultralytics import YOLO                         # Ultralytics YOLO（v8）检测框架
import cv2                                           # OpenCV：图像处理与显示
import threading                                     # 线程库：后台线程负责读视频
import numpy as np                                   # 数值计算：用于构造 HSV 阈值数组、形态学核
import time                                          # 时间库：控制读帧节奏


class VisionPerceptionNode(Node):                    # 视觉感知节点：读视频 -> 传统 CV + YOLO 检测 -> 发布并显示
    def __init__(self):
        super().__init__('vision_perception_node')   # 初始化父类，节点名 vision_perception_node

        # 加载 YOLOv8n 预训练权重（相对路径，必须在 OpenCv 目录下启动才找得到）
        self.model = YOLO('hsv_heixin/kaohei/kaoheiku_ws/yolov8n.pt')
        self.conf_thres = 0.5        # 置信度阈值：低于 0.5 的检测结果会被丢弃
        self.device = 'cpu'          # 推理设备，可改为 'cuda' 使用 GPU
        self.bridge = CvBridge()     # 图像转换器，实例化一次重复使用
        # 创建发布者：把画好框的图像发布到 /perception/annotated_image，队列深度 10
        self.image_pub = self.create_publisher(Image, '/perception/annotated_image', 10)

        # 视频源：用本地 mp4 代替实时摄像头，方便离线调试
        self.video_path = '/home/jianmo/OpenCv/hsv_heixin/kaohei/kaoheiku_ws/shopi.mp4'
        self.cap = cv2.VideoCapture(self.video_path)        # 打开视频文件/摄像头
        if not self.cap.isOpened():                        # 打开失败（路径错误或编码不支持）
            self.get_logger().error(f'视频打开失败: {self.video_path}')
            raise SystemExit                               # 直接终止构造，避免后续拿到无效句柄

        # 读取视频元信息
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)                          # 帧率
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))    # 总帧数
        self.get_logger().info(f'视频已加载 | FPS={self.fps:.2f} | 总帧数={self.total_frames}')

        # 预先创建 OpenCV 窗口，WINDOW_NORMAL 表示可自由拖动缩放
        cv2.namedWindow('1. Original', cv2.WINDOW_NORMAL)       # 窗口1：原始帧（本行先创建，下面再统一设尺寸）
        cv2.namedWindow('OpenCV Pipeline', cv2.WINDOW_NORMAL)   # 预留窗口，本文件中并未使用
        cv2.resizeWindow('1. Original', 640, 360)               # 设置窗口1尺寸 640x360
        cv2.namedWindow('2. Gray', cv2.WINDOW_NORMAL)           # 窗口2：灰度图
        cv2.resizeWindow('2. Gray', 640, 360)                   # 设置窗口2尺寸
        cv2.namedWindow('3. Blurred', cv2.WINDOW_NORMAL)        # 窗口3：高斯模糊结果
        cv2.resizeWindow('3. Blurred', 640, 360)                # 设置窗口3尺寸
        cv2.namedWindow('4. Edges', cv2.WINDOW_NORMAL)          # 窗口4：Canny 边缘图
        cv2.resizeWindow('4. Edges', 640, 360)                  # 设置窗口4尺寸

        # 读取视频分辨率并再次打印完整信息
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # 帧宽度
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # 帧高度
        self.get_logger().info(f'宽度={width} | 高度={height} | FPS={self.fps:.2f} | 总帧数={self.total_frames}')

        self.running = True                                                  # 运行标志（当前代码里只写不读，属冗余状态）
        self.frame_lock = threading.Lock()                                   # 线程锁：保护 latest_frame 的读写
        self.latest_frame = None                                             # 最新一帧：读线程写、处理线程读
        self.reader_thread = threading.Thread(target=self._video_reader, daemon=True)  # 后台读帧线程，daemon 随主进程退出
        self.reader_thread.start()                                           # 启动读帧线程
        # 创建定时器：按视频帧率周期调用 _process_frame（fps 无效时保底 1Hz）
        self.timer = self.create_timer(1.0 / max(self.fps, 1.0), self._process_frame)

    def _video_reader(self):
        # 后台线程：不停从视频读帧，只保留最新一帧，避免处理慢时帧堆积
        while self.running:             
            ret, frame = self.cap.read()     # 读一帧，ret=False 表示读完或读取失败
            if not ret:
                self.get_logger().info('视频读取完毕')
                self.running = False         # 标记已结束（实际没有逻辑读取该标志）
                break                        # 跳出读帧循环
            with self.frame_lock:            # 加锁写入，防止处理线程读到写了一半的帧引用
                self.latest_frame = frame    # 覆盖保存最新帧（旧帧直接丢弃）
            time.sleep(1.0 / max(self.fps, 1.0))   # 按帧率休眠，模拟实时播放节奏

    def _process_frame(self):
        # 定时器回调：取最新帧 -> 传统 CV 处理 -> YOLO 检测 -> HSV 红框 -> 发布/显示
        with self.frame_lock:                     # 加锁读取，与写线程互斥
            if self.latest_frame is None:         # 读线程还没产出第一帧
                return                            # 本次直接跳过
            frame = self.latest_frame.copy()      # 拷贝一份，避免后续绘制污染共享帧

        # ---------- 传统 OpenCV 处理流水线 ----------
        cv2.imshow('1. Original', frame)                     # 原图：直接显示 BGR 帧

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       # 灰度：BGR -> 单通道灰度
        cv2.imshow('2. Gray', gray)                          # 显示灰度图

        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)        # 高斯：5x5 核、sigma=1.5 平滑去噪
        cv2.imshow('3. Blurred', blurred)                    # 显示模糊结果

        edges = cv2.Canny(blurred, 50, 150)                  # 边缘：Canny 双阈值，低 50 / 高 150
        cv2.imshow('4. Edges', edges)                        # 显示边缘图

        # ---------- YOLOv8 目标检测 ----------
        # 传入 BGR 帧、置信度阈值、设备，verbose=False 关闭逐帧日志
        results = self.model(frame, conf=self.conf_thres, device=self.device, verbose=False)
        result = results[0]        # 取第一张（也是唯一一张）图的检测结果
        annotated = result.plot()  # 把检测框、类别、置信度画到图上，返回 BGR 图像

        # ---------- 基于 HSV 的红色目标检测 ----------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)      # BGR -> HSV，便于按颜色分割
        # 红色在色相环上跨越 0 度，需要两段阈值分别覆盖
        mask1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))     # 低色相段 H∈[0,10]
        mask2 = cv2.inRange(hsv, np.array([160, 100, 80]), np.array([179, 255, 255]))  # 高色相段 H∈[160,179]
        mask = cv2.bitwise_or(mask1, mask2)               # 两段掩膜按位或，得到完整红色区域
        kernel = np.ones((5, 5), np.uint8)                # 5x5 形态学结构元素
        # 先开运算（去小噪点）再闭运算（填孔洞），闭运算迭代 2 次
        mask_closed = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel, iterations=2)

        # 只提取最外层轮廓（掩膜中每个红色色块的外边界）
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:                              # 遍历每个轮廓
            area = cv2.contourArea(cnt)                   # 计算轮廓面积（像素数）
            if 800 < area < 500000:                       # 面积过滤：去掉小噪点和过大的背景块
                x, y, w, h = cv2.boundingRect(cnt)        # 求最小正矩形（左上角 + 宽高）
                if h > 0 and 0.2 < (w / float(h)) < 5.0:  # 宽高比过滤，排除细长条误检
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)   # 在标注图上画绿框，线宽 2

        # ---------- 将标注结果发布为 ROS 话题 ----------
        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')   # OpenCV 图像 -> ROS Image 消息
            img_msg.header.stamp = self.get_clock().now().to_msg()            # 写入当前 ROS 时间戳
            img_msg.header.frame_id = 'camera'                               # 标注坐标系名，便于 TF 关联
            self.image_pub.publish(img_msg)                                  # 发布到 /perception/annotated_image
        except Exception as e:
            self.get_logger().error(f'图像发布失败: {e}')                      # 发布失败只记日志，不中断主循环

        cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)   # 创建/复用检测结果窗口（放在回调里创建，每帧重复调用）
        cv2.resizeWindow('YOLO Detection', 960, 540)           # 设置窗口尺寸 960x540
        cv2.imshow('YOLO Detection', annotated)                # 显示带 YOLO 框 + 红色框的图像

        key = cv2.waitKey(1) & 0xFF        # 处理窗口事件 1ms，并取按键码的低 8 位
        if key == ord('q') or key == 27:   # 按下 q 或 Esc（ASCII 27）则退出
            self.get_logger().info('用户按键退出')
            self.running = False           # 置退出标志
            rclpy.shutdown()               # 关闭 ROS 上下文

            
    def destroy_node(self):
        self.running = False                         # 先置结束标志，通知后台线程退出
        if self.reader_thread.is_alive():            # 读帧线程仍在运行
            self.reader_thread.join(timeout=1.0)     # 最多等待 1 秒，防止线程卡住退出流程
        if self.cap.isOpened():                      # 视频句柄仍然打开
            self.cap.release()                       # 释放视频文件/摄像头资源
        cv2.destroyAllWindows()                      # 关闭全部 OpenCV 窗口
        super().destroy_node()                       # 调用父类销毁逻辑，释放 ROS 资源


def main(args=None):
    rclpy.init(args=args)             # 初始化 ROS 2 上下文
    node = VisionPerceptionNode()     # 创建节点（构造函数中已开始读视频、创建定时器）
    try:
        rclpy.spin(node)              # 事件循环：驱动定时器回调 _process_frame
    except KeyboardInterrupt:         # 捕获 Ctrl+C
        pass                          # 忽略异常，交给 finally 统一清理
    finally:
        node.destroy_node()           # 释放视频、窗口与 ROS 资源
        if rclpy.ok():                # Ctrl+C 时 rclpy 可能已自行关闭上下文，避免重复 shutdown 抛 RCLError
            rclpy.shutdown()          # 关闭 ROS 2 上下文，与 init 成对


if __name__ == '__main__':     # 仅当直接运行本文件时生效（被 import 时不启动）
    main()                     # 进入主函数
