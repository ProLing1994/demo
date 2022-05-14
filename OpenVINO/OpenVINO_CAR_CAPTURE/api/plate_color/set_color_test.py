import cv2
import numpy as np

# 可视化实验参考：https://blog.csdn.net/Aiden_yan/article/details/117134039
def nothing(x):
    pass

cv2.namedWindow('result')
cv2.resizeWindow('result', 600, 800)

# 设置hsv6个分量，分别为low_H,high_H,low_S,high_S,low_V,high_V
'''
cv2.createTrackbar参数解释：
第一个参数时滑动条的名字，
第二个参数是滑动条被放置的窗口的名字，
第三个参数是滑动条默认值，
第四个参数时滑动条的最大值，
第五个参数时回调函数，每次滑动都会调用回调函数。
'''
cv2.createTrackbar("Low_H", "result", 90, 255, nothing)
cv2.createTrackbar("High_H", "result", 124, 255, nothing)
cv2.createTrackbar("Low_S", "result", 43, 255, nothing)
cv2.createTrackbar("High_S", "result",255, 255, nothing)
cv2.createTrackbar("Low_V", "result", 46, 255, nothing)
cv2.createTrackbar("High_V", "result", 255, 255, nothing)

while True:
    # image_path = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_ahhbgs_5M/yellow/沪DS3228_0.jpg"
    # image_path = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_zhjyz_2M/green/粤BA95561_764.jpg"
    image_path = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_zhjyz_2M/blue/湘J55N30_2828.jpg"
    img = cv2.imread(image_path)
    resize = cv2.resize(img, None, fx=0.5, fy=0.5)
    hsv = cv2.cvtColor(resize, cv2.COLOR_BGR2HSV)

    # 分别获取被拖动后的HSV值
    '''
    cv2.getTrackbarPos()参数解释：
    第一个参数为滚动条的名称
    第二个参数为窗口的名字
    '''
    low_h = cv2.getTrackbarPos('Low_H', 'result')
    hign_h = cv2.getTrackbarPos('High_H', 'result')
    low_s = cv2.getTrackbarPos('Low_S', 'result')
    hign_s = cv2.getTrackbarPos('High_S', 'result')
    low_v = cv2.getTrackbarPos('Low_V', 'result')
    hign_v = cv2.getTrackbarPos('High_V', 'result')

    # 分别设置高低阈值
    low_threshold = np.array([low_h, low_s, low_v])
    high_threshold = np.array([hign_h, hign_s, hign_v])

    # 根据阈值构建掩膜
    mask = cv2.inRange(hsv, low_threshold, high_threshold)

    # 对原图像和掩膜进行位运算
    res = cv2.bitwise_and(resize, resize, mask=mask)

    cv2.imshow("resize", resize)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    if cv2.waitKey(1)== 27:  # 按Esc退出,cv2.waitKey(x)里面的数不能为0
        break

cv2.destroyAllWindows()