import cv2                                    # 导入 OpenCV：读画面、转色彩空间、做阈值、显示窗口
import numpy as np                            # 导入 NumPy：用数组保存 HSV 上下界


def nothing(x):                               # 滑块回调函数（createTrackbar 必须传一个函数，这里什么都不做）
    pass                                      # 空实现：滑块拖动时不需要额外动作


cap = cv2.VideoCapture(0)                     # 打开 0 号摄像头（没有摄像头时 isOpened() 为 False，循环会立刻结束）
cv2.namedWindow('trackbar')                   # 创建名为 trackbar 的窗口，用来放滑块

cv2.createTrackbar('H_min', 'trackbar', 0, 179, nothing)      # 色相下限滑块：范围 0~179，默认 0
cv2.createTrackbar('H_max', 'trackbar', 179, 179, nothing)    # 色相上限滑块：范围 0~179，默认 179
cv2.createTrackbar('S_min', 'trackbar', 0, 255, nothing)      # 饱和度下限滑块：范围 0~255，默认 0
cv2.createTrackbar('S_max', 'trackbar', 255, 255, nothing)    # 饱和度上限滑块：范围 0~255，默认 255
cv2.createTrackbar('V_min', 'trackbar', 0, 255, nothing)      # 明度下限滑块：范围 0~255，默认 0
cv2.createTrackbar('V_max', 'trackbar', 255, 255, nothing)    # 明度上限滑块：范围 0~255，默认 255

while True:                                   # 主循环：每轮处理一帧，直到退出
    ret, frame = cap.read()                   # 读一帧；ret 表示是否读成功，frame 是 BGR 图像
    if not ret:                               # 没读到（摄像头没打开或视频播完）
        break                                 # 跳出循环，结束程序
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # BGR 转 HSV，便于按颜色阈值分割

    h_min = cv2.getTrackbarPos('H_min', 'trackbar')   # 读取 H_min 滑块当前值
    h_max = cv2.getTrackbarPos('H_max', 'trackbar')   # 读取 H_max 滑块当前值
    s_min = cv2.getTrackbarPos('S_min', 'trackbar')   # 读取 S_min 滑块当前值
    s_max = cv2.getTrackbarPos('S_max', 'trackbar')   # 读取 S_max 滑块当前值
    v_min = cv2.getTrackbarPos('V_min', 'trackbar')   # 读取 V_min 滑块当前值
    v_max = cv2.getTrackbarPos('V_max', 'trackbar')   # 读取 V_max 滑块当前值

    lower = np.array([h_min, s_min, v_min])   # 阈值下界数组 [H,S,V]
    upper = np.array([h_max, s_max, v_max])   # 阈值上界数组 [H,S,V]
    mask = cv2.inRange(hsv, lower, upper)     # 阈值内的像素变白(255)，其余变黑(0)，得到二值掩膜
    result = cv2.bitwise_and(frame, frame, mask=mask)  # 用掩膜抠图：只保留命中颜色的部分

    cv2.imshow('original', frame)             # 窗口1：原始 BGR 画面，用来对照目标
    cv2.imshow('mask', mask)                  # 窗口2：二值掩膜，主要看这个判断阈值是否合适
    cv2.imshow('result', result)              # 窗口3：抠图结果，确认目标颜色是否完整

    if cv2.waitKey(1) & 0xFF == 27:           # 处理窗口事件 1ms；按键码等于 27（Esc）则退出
        break                                 # 跳出循环

cap.release()                                 # 释放摄像头资源
cv2.destroyAllWindows()                       # 关闭所有 OpenCV 窗口
