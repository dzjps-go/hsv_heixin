import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jianmo/OpenCv/hsv_heixin/vision_ws/install/vision_tutorial'
