import cv2, numpy as np, time

def get_perspective_mat():
    src_points = np.array([[255, 1000], [1725, 1000], [540, 380], [1400, 380]], dtype="float32")
    dst_points = np.array([[255, 1000], [1725, 1000], [255, 0], [1725, 0]], dtype="float32")

    return cv2.getPerspectiveTransform(src_points, dst_points)

cam = cv2.VideoCapture("http://192.168.137.113:4747/video")
# time.sleep(3)
# ret, img = cam.read()
# cv2.imwrite("test.jpg", img)
while True:
    ret, raw_img = cam.read()
    img = raw_img.copy()

    # IPM
    ipm = cv2.warpPerspective(img, get_perspective_mat(), (480, 270), cv2.INTER_LINEAR)

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 紅色 0 181 204
    # lower_green = np.array([-10, 140, 0])
    # upper_green = np.array([10, 255, 255])

    # 藍色 155 255 125
    # lower_green = np.array([100, 125, 100])
    # upper_green = np.array([120, 255, 255])

    # 黃色 37 255 204
    lower_green = np.array([100, 125, 100])
    upper_green = np.array([120, 255, 255])

    # 建立遮罩
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    col_out = cv2.bitwise_and(img, img, mask=mask)

    # if contours have been detected, draw them
    if len(contours) > 0:

        # 繪製邊框
        cv2.drawContours(img, contours, -1, 255, 2)

        # 取得最大面積
        largestContour = np.array([[]])
        llpython = [0,0,0,0,0,0,0,0]
        largestContour = max(contours, key=cv2.contourArea)

        # 取得目標外框大小
        x,y,w,h = cv2.boundingRect(largestContour)

        # 大小閥值判斷
        if len(largestContour) > 40 and len(largestContour) < 750:
            # 繪製目標外框
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 2)

        # record some custom data to send back to the robot
        llpython = [1,x,y,w,h,9,8,7]

    # cv2.imshow("IPM", ipm)
    cv2.imshow("Origin", img)
    cv2.imshow("IPM", ipm)
    cv2.imshow("HSV", hsv)
    cv2.imshow("Color", col_out)

    if cv2.waitKey(5) == 27:
        break
