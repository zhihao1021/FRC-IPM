import cv2, numpy as np, time

def get_perspective_mat():
    src_points = np.array([[80, 480], [560, 480], [190, 220], [450, 220]], dtype="float32")
    dst_points = np.array([[80, 480], [560, 480], [80, 0], [560, 0]], dtype="float32")

    return cv2.getPerspectiveTransform(src_points, dst_points)

def det(img, upper, lower):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        cv2.drawContours(img, contours, -1, 255, 2)
        largestContour = np.array([[]])
        largestContour = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largestContour)
        if len(largestContour) > 40 and len(largestContour) < 750:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 2)
    return img

cam = cv2.VideoCapture("http://192.168.137.113:4747/video")
# time.sleep(3)
# ret, img = cam.read()
# cv2.imwrite("test.jpg", img)
while True:
    ret, raw_img = cam.read()
    img = raw_img.copy()

    # IPM
    ipm = cv2.warpPerspective(img, get_perspective_mat(), (640, 480), cv2.INTER_LINEAR)

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 紅色 0 181 204
    # lower = np.array([-10, 140, 0])
    # upper = np.array([10, 255, 255])

    # 藍色 155 255 125
    lower = np.array([100, 125, 100])
    upper = np.array([120, 255, 255])

    # 黃色 37 255 204
    # lower = np.array([100, 125, 100])
    # upper = np.array([120, 255, 255])

    det_img = det(img.copy(), upper, lower)
    ipm_img = det(ipm, upper, lower)

    cv2.imshow("Origin", img)
    cv2.imshow("IPM", ipm)
    cv2.imshow("HSV", hsv)
    cv2.imshow("Det", det_img)

    if cv2.waitKey(5) == 27:
        break
