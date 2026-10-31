import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jianmo/OpenCv/hsv_heixin/kaohei/kaoheiku_ws/install/vision_perception'
