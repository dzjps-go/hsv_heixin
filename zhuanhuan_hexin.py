import cv2  # 导入 OpenCV 库，用于读取图片和颜色空间转换
import numpy as np  # 导入 NumPy 库，用于数值与数组运算（本脚本中暂时未使用，但便于后续扩展）

img_bgr = cv2.imread('test.jpg')  # 读取当前目录下的 test.jpg 图片，默认以 BGR 三通道形式读入（numpy 数组）
img_hsv = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)  # 把 BGR 图像转换为 HSV 颜色空间（H 色相/S 饱和度/V 明度）
print(img_hsv.shape)  # 打印图像数组的形状 (高, 宽, 通道数)，用于确认读取和转换是否成功
