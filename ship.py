import cv2
import rclpy
from rclpy.node import Node
import numpy as np

#原图
img = cv2.imread('/home/jianmo/OpenCv/hsv_heixin/Open_Cv/tuoian1.jpg')
print(f"形状",{img.shape})
print(f'数据类型',{img.dtype})
print(f'图形大小',{img.size})
roi = img[30:400,50:300]
cv2.imshow('tu',img)
cv2.imshow('qie',roi)
cv2.waitKey(1000)
cv2.destroyAllWindows()

#灰图
gray = cv2.imread('/home/jianmo/OpenCv/hsv_heixin/Open_Cv/tuoian1.jpg',cv2.IMREAD_GRAYSCALE)
print(f"形状",{gray.shape})
print(f'数据类型',{gray.dtype})
print(f'图形大小',{gray.size})
cv2.imshow('hui',gray)
cv2.waitKey(1000)
cv2.destroyAllWindows()


#视频
video = cv2.VideoCapture('/home/jianmo/OpenCv/shipii.mp4')
if not video.isOpened():
    print('无法打开此视频')
    exit()

while True:
    ret,farm = video.read()
    if not ret:
        break

    cv2.imshow('video',farm)
    if cv2.waitKey(6) ==  27: #cv2.waitKey(60)是控制视频速度的
        break

video.release()
cv2.destroyAllWindows()

#颜色通道分离
yanshe = cv2.imread('/home/jianmo/OpenCv/hsv_heixin/Open_Cv/tupian2.jpg')

b = yanshe[:,:,0]
g = yanshe[:,:,1]
r = yanshe[:,:,2]

b , g , r = cv2.split(yanshe)
cv2.imshow('b',b)
cv2.imshow('g',g)
cv2.imshow('r',r)
cv2.waitKey(1100)
cv2.destroyAllWindows()

#合并
merged = cv2.merge((b,g,r))
cv2.imshow('bgr',merged)
cv2.waitKey(100)
cv2.destroyAllWindows()

#指定地区打码
mask = cv2.imread('/home/jianmo/OpenCv/hsv_heixin/Open_Cv/tuoian1.jpg')
mask[100:200,200:300] = np.random.randint(0,256,(100,100,3))
cv2.imshow('dama',mask)
cv2.waitKey(100)
cv2.destroyAllWindows()

#图像组合
zhu = cv2.imread('/home/jianmo/OpenCv/hsv_heixin/Open_Cv/tuoian1.jpg')
he = cv2.imread('/home/jianmo/OpenCv/hsv_heixin/Open_Cv/tupian2.jpg')

he[200:350,200:350] = zhu[100:250,300:450]

cv2.imshow("zhuhe",he)
cv2.waitKey(1000)
cv2.destroyAllWindows()

#缩放
shuofang = cv2.imread('/home/jianmo/OpenCv/hsv_heixin/Open_Cv/tupian2.jpg')
ili = cv2.resize(shuofang,None,fx=2,fy=0.5)
cv2.imshow('shuo',shuofang)
cv2.imshow('ili',ili)
cv2.waitKey(10000)
cv2.destroyAllWindows()

#改图
gai = cv2.imread('/home/jianmo/OpenCv/hsv_heixin/Open_Cv/tuoian1.jpg')
gai = cv2.resize(gai,(500,500))
hui = cv2.imread('/home/jianmo/OpenCv/hsv_heixin/Open_Cv/tupian2.jpg',cv2.IMREAD_GRAYSCALE)
hui = cv2.resize(hui,(500,500))
gai[250:500,250:400] = gai[0:250,100:250]
gai[50:140,320:410] = np.random.randint(0.256,(90,90,3))
cv2.imshow('yuan',gai)
cv2.imshow('hui',hui)
cv2.waitKey(10000)
cv2.destroyAllWindows()