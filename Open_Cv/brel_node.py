import cv2

path = '/home/jianmo/OpenCv/hsv_heixin/Open_Cv/tuoian1.jpg'

img = cv2.imread(path, cv2.IMREAD_COLOR)
if img is None:
    raise SystemExit(f'图片读取失败，请检查路径: {path}')

gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

COLOR_WIN = 'color'
GRAY_WIN = 'gray'

cv2.namedWindow(COLOR_WIN, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(GRAY_WIN, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(COLOR_WIN, 50, 50)
cv2.moveWindow(GRAY_WIN, 50 + img.shape[1] + 40, 50)

cv2.imshow(COLOR_WIN, img)
cv2.imshow(GRAY_WIN, gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
