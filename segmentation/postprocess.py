import cv2


def post_process(img):
    img2 = img
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contours, -1, (255, 0, 0))
    print(len(contours))
    # print(cv2.contourArea(contours[0]))
    for i in range(len(contours)):
        # print(cv2.contourArea(contours[i]))
        if cv2.contourArea(contours[i]) < 2000:
            cv2.drawContours(img, [contours[i]], 0, 1, -1)


    return img
