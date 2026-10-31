import os
os.environ["PYTORCH_JIT"] = "0"
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "local"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import threading
import numpy as np
import time



class VisionPerceptionNode(Node):
    def __init__(self):
        super().__init__('vision_perception_node')


        self.model = YOLO('hsv_heixin/kaohei/kaoheiku_ws/yolov8n.pt')          
        self.conf_thres = 0.5                     
        self.device = 'cpu'                    
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/perception/annotated_image', 10)




        self.video_path = 'hsv_heixin/kaohei/kaoheiku_ws/shopi.mp4'  
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'视频打开失败: {self.video_path}')
            raise SystemExit
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.get_logger().info(f'视频已加载 | FPS={self.fps:.2f} | 总帧数={self.total_frames}')



        cv2.namedWindow('1. Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('OpenCV Pipeline', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('1. Original', 640, 360)
        cv2.namedWindow('2. Gray', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('2. Gray', 640, 360)
        cv2.namedWindow('3. Blurred', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('3. Blurred', 640, 360)
        cv2.namedWindow('4. Edges', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('4. Edges', 640, 360)





        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f'宽度={width} | 高度={height} | FPS={self.fps:.2f} | 总帧数={self.total_frames}')
        self.running = True
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.reader_thread = threading.Thread(target=self._video_reader,daemon=True)
        self.reader_thread.start()
        self.timer = self.create_timer(1.0 / max(self.fps, 1.0), self._process_frame)




    def _video_reader(self):
        while self.frame_lock:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().info('视频读取完毕')
                self.running = False
                break
            with self.frame_lock:
                self.latest_frame = frame
            time.sleep(1.0 / max(self.fps, 1.0))


    def _process_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()




        #原图
        cv2.imshow('1. Original', frame)


        #灰图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('2. Gray', gray)


        #高斯
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        cv2.imshow('3. Blurred', blurred)


        #边缘测
        edges = cv2.Canny(blurred, 50, 150)
        cv2.imshow('4. Edges', edges)





        results = self.model(frame, conf=self.conf_thres, device=self.device, verbose=False)
        result = results[0]
        annotated = result.plot()  


        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 100, 80]), np.array([179, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel, iterations=2)


        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 800 < area < 500000: 
                x, y, w, h = cv2.boundingRect(cnt)
                if h > 0 and 0.2 < (w / float(h)) < 5.0: 
                  cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2) 




        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'camera'
            self.image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'图像发布失败: {e}')




        cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO Detection', 960, 540)
        cv2.imshow('YOLO Detection', annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            self.get_logger().info('用户按键退出')
            self.running = False
            rclpy.shutdown()




    def destroy_node(self):
        self.running = False
        if self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()



def main(args=None):
    rclpy.init(args=args)
    node = VisionPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()