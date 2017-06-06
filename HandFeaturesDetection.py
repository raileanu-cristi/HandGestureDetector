import cv2
import numpy as np
import math

def getConvexPointsInCoutour(img, binImage):
    # print(np.median(binImage))
    # print(binImage.shape)
    # print(binImage.dtype)
    _, contours, hierarchy = cv2.findContours(binImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # convexity defects
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html#contours-more-functions   

    maxArea = 0
    maxInd = -1
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > maxArea:
            maxArea, maxInd = area, i

    # print(len(contours), ' ',maxInd )
    if maxInd == -1:
        return img

    cnt = contours[maxInd]

    cv2.drawContours(img, contours, contourIdx=maxInd, color=(0, 255, 0), thickness=1)

    # x,y,w,h = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    # cv2.drawContours(img, [hull], contourIdx=0, color=(255, 0, 0), thickness=1) # should be active to see more

    # convexity defects
    #
    hull = cv2.convexHull(cnt, returnPoints = False)    
    defects = cv2.convexityDefects(cnt,hull)

    if defects==None:
        return img

    fingerNr = 1

    # http://creat-tabu.blogspot.ro/2013/08/opencv-python-hand-gesture-recognition.html
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # filter defects
        v1 = np.array(start) - np.array(far)
        v2 = np.array(end) - np.array(far)
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) )
        teta = math.acos(cos)
        teta = teta / math.pi * 180 # in degrees
        # print( teta )
        if teta > 100 or teta < 10:
            continue

        fingerNr += 1
        cv2.circle(img,start,4,[255,255,0],-1)
        # end
        cv2.circle(img,end,4,[255,255,0],-1)
        # far
        cv2.circle(img,far,4,[0,0,255],-1)

        # dist = cv2.pointPolygonTest(cnt,centr,True)
    #     cv2.line(img,start,end,[0,255,0],2)                
    #     cv2.circle(img,far,5,[0,0,255],-1)
    # exit() # debug
    # print(defects)
    return img, fingerNr



def getFeatures(img, binImage):

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	face_cascade = cv2.CascadeClassifier('E:\Programe\openCV\opencv\data\haarcascades\haarcascade_frontalface_default.xml')

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		binImage[ y:y+int(h*1.5), x:x+w ] = 0

	_, contours, hierarchy = cv2.findContours(binImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	maxArea, maxInd = 0, -1
	for i in range(0, len(contours)):
		area = cv2.contourArea(contours[i])
		if area > maxArea:
			maxArea, maxInd = area, i

	# print(len(contours), ' ',maxInd )
	if maxInd == -1:
		return img

	cnt = contours[maxInd]

	cv2.drawContours(img, contours, contourIdx=maxInd, color=(0, 255, 0), thickness=1)

	(x,y),radius = cv2.minEnclosingCircle(cnt)
	center = (int(x),int(y))
	radius = int(radius)
	cv2.circle(img,center,radius,(0,255,0),2)

	rect = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(img,[box],0,(0,0,255),2)
	
	return img, 0