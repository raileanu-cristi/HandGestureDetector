import cv2
import numpy as np
import math


class DetectionGlobalParams:
	minArea = 100
	polygonAproximationError = 0.001
	nPoint = 1
	kCurvatureRadius = 20
	kCurvatureK = 30
	roiRadiusScale = 2.8


def runDetection(binaryImage):

	contours = contourExtraction(binaryImage, DetectionGlobalParams.minArea, DetectionGlobalParams.polygonAproximationError )

	maxCircles = listAdapter(maxInscribedCircle, contours, (DetectionGlobalParams.nPoint, binaryImage.shape))
	# print('maxInscribedCircle')
	rafinedBinaryImage = regionsOfInterest(binaryImage, maxCircles, DetectionGlobalParams.roiRadiusScale)
	# print('regionsOfInterest')
	
	contours = contourExtraction(rafinedBinaryImage, DetectionGlobalParams.minArea, DetectionGlobalParams.polygonAproximationError )
	# print('contourExtraction')
	contoursInfo = listAdapter(processContour, contours, 0)
	# print('processContour')
	handsFingerInfo = listAdapter(fingerInformation, zip(contours, contoursInfo, maxCircles), 0 )
	# print('fingerInformation')
	return contours, maxCircles, handsFingerInfo



def listAdapter(f, listArg, params):
	results = []
	for element in listArg:
		results += [f(element, params)]
	return results



def contourExtraction(binaryImage, threshAreaValue, allowedError):

	_, contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# contour threshhold
	chosenContours = []
	for i in range(0, len(contours)):
		area = cv2.contourArea(contours[i])
		if area >= threshAreaValue:
			chosenContours += [contours[i]]
	# contour aproximation
	for i in range(0, len(chosenContours)):
		cnt = chosenContours[i]
		epsilon = allowedError * cv2.arcLength(cnt,True) # the distance to remain (% of perimeter)
		chosenContours[i] = cv2.approxPolyDP(cnt,epsilon,True)

	return chosenContours



def maxInscribedCircle(contour, params):
	nPoint, frameShape = params
	# downscale the image
	# limit ROI to contour bounding rectangle

	x,y,w,h = cv2.boundingRect(contour)
	x, y = x + w/5*2, y + h/5*2 
	w, h = w/5, h/5
	rectangle = [(x,y), (x+w, y), (x+w, y+h), (x, y+h) ]	

	points = pointsInRectangle(rectangle, frameShape)
	
	# search for every N-point (N = 4)
	center = (0, 0)
	radius = 0
	for newCenter in points:
		
		minDist = 10000
		i = 0

		while i<len(contour) :
			# compute min distance
			point = contour[i]
			dist = distance( newCenter, point )
			minDist = min(minDist, dist)
			
			i += nPoint

		if minDist > radius:
			center = int(newCenter[0]), int(newCenter[1])
			radius = int(minDist)
		
	maxCircle =(center, radius)

	return maxCircle



def pointsInRectangle(rectangle, frameShape):

	x1, y1 = rectangle[0]
	x2, y2 = rectangle[1]
	x4, y4 = rectangle[3]

	vect1 = np.array([x2-x1, y2-y1])
	vect2 = np.array([x4-x1, y4-y1])
	dist1 = np.linalg.norm(vect1)
	dist2 = np.linalg.norm(vect2)
	k = 3
	smallVect1 = vect1 / dist1 * k
	smallVect2 = vect2 / dist2 * k

	eps = 4

	points = []
	origin = np.array([x1, y1])
	farOrigin = np.array([x2, y2])
	currentFirstPoint = np.array([x1, y1])

	while ( distance( tuple(origin) , tuple(currentFirstPoint)  ) < dist2 ):
		currentPoint = np.array(currentFirstPoint)
		# print (distance( tuple(currentPoint), tuple( currentFirstPoint )), ' ', dist2)

		while ( distance( tuple(currentPoint), tuple( currentFirstPoint )  ) < dist1):
			# print (distance( tuple(currentPoint), tuple( currentFirstPoint )), ' ', dist1)
			if np.all(currentPoint >= np.zeros(2)) and np.all(currentPoint <= np.array(frameShape)):
				points += [ tuple(np.int_(currentPoint)) ]
				# print(currentPoint)

			currentPoint += smallVect1
			# print (len(points))
		currentFirstPoint += smallVect2
	
	return points



def regionsOfInterest(image, circles, scaleNr):
	
	newImage = np.zeros(image.shape, dtype= np.uint8)

	# print(circles)
	for circle in circles:
		print(circle)
		center, radius = circle
		x, y = center
		radius = int(radius * scaleNr)
		print(radius)
		print(x,' ',y,' ',radius)
		x1, y1 = max(1, x - radius), max( 1, y - radius)
		x2, y2 = min(image.shape[0], x + radius), min(image.shape[1], y + radius)
		newImage[x1:x2, y1:y2] = image[x1:x2, y1:y2 ]

	return newImage



def processContour(contour, params):
	hull = cv2.convexHull(contour,returnPoints = False)
	defects = cv2.convexityDefects(contour,hull)
	
	angle = 0 			# TODO K-curvature
	minEnclosingCircle = cv2.minEnclosingCircle(contour)

	return hull, defects, minEnclosingCircle



def fingerInformation(handInfo, params):

	contour,(hull, defects, minEnclosingCircle), maxCirle = handInfo

	center, ra = maxCirle
	center, rb = minEnclosingCircle

	convexityDefects = []
	Ap = []
	for i in range(defects.shape[0]):
		s,e,f,ld = defects[i,0]
		pStart = tuple(contour[s][0])
		pEnd = tuple(contour[e][0])
		pFar = tuple(contour[f][0])

		if ld >= ra and ld <=rb and angle(pFar, pStart, pEnd) < 90:
			if len(Ap) == 0:
				Ap += s
			Ap += [e, f]
	
	print(Ap)
	kCurvaturesInfo = listAdapter(kCurvature, Ap, (contour, 20, 30))

	return kCurvaturesInfo



def kCurvature(pos, params):

	contour, radius, k = params

	peakPoint, middlePoint, minAngle = (0, 0), (0, 0) , 180

	nrContourPoints = len(contour)
	for i in range(-radius, radius):
		originPos = (pos + i + nrContourPoints) % nrContourPoints
		leftPos = (originPos - k + nrContourPoints) % nrContourPoints
		rightPos = (originPos + k + nrContourPoints) % nrContourPoints
		origin = tuple(contour[originPos][0])
		left = tuple(contour[leftPos][0])
		right = tuple(contour[rightPos][0])

		teta = angle(origin, left, right)
		
		if teta < minAngle:
			minAngle = teta
			peakPoint = origin
			middlePoint = (np.int(left) + np.int(right)) / 2

	return peakPoint, middlePoint, minAngle



def angle(origin, p1, p2):
	v1 = np.array(p1) - np.array(origin)
	v2 = np.array(p2) - np.array(origin)
	cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) )
	teta = math.acos(cos)
	teta = teta / math.pi * 180 # in degrees
	return teta



def distance(p1, p2):
	v1 = np.array(p1) - np.array(p2)
	return np.linalg.norm(v1) 



def mainTest():
	imgInput = cv2.imread('binary_hand.bmp')
	# cv2.imshow('img', img)
	# if cv2.waitKey(0) == ord('q'):
 #            exit()

	imgInput = cv2.cvtColor(imgInput,cv2.COLOR_BGR2GRAY)
	handsFingerInfo = runDetection(imgInput)

	# print(maxCircle)
	contours, maxCircle, fingerInfo = handsFingerInfo
	contour = contours[0]
	center, radius = maxCircle[0]
	frameShape = imgInput.shape

	img = np.zeros( (frameShape[0], frameShape[1], 3))
	cv2.drawContours(img,[contour],0,(255,255,255),2)
	cv2.circle(img,center, radius, (0,255,0), 0)
	cv2.circle(img, center, 5, (0,255,255), -1  )
	# cv2.drawContours(img,[ np.array(rectangle, dtype=np.int0)],0,(255,0,0),2)
	
	# print(fingerInfo)
	# peakPoint, middlePoint, angle = fingerInfo[0]
	# cv2.line(img,peakPoint,middlePoint,(255,0,0),3)

	cv2.imshow('img', img)
	cv2.waitKey(0)

	return

mainTest()


# deteled code
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	# lst = []
	# for i in range(0, 20*1000*1000):
	# 	lst += [i]
	# lst = np.zeros(20*1000*1000)
	# for i in range(0, 20*1000*1000):
	# 	lst[i] = i

	# rect = cv2.minAreaRect(contour)
	# box = cv2.boxPoints(rect)
	# print(box)	# debug
	# box = np.int0(box)

	# # print(box)	# debug
	
	# cv2.drawContours(img,[box],0,(0,0,255),2)
	# # reduce the width of the box by 3x
	# x1, y1 = box[0]
	# x2, y2 = box[1]
	# x4, y4 = box[3]
	# vect1 = np.array([x2-x1, y2-y1])
	# vect2 = np.array([x4-x1, y4-y1])
	# dist1 = np.linalg.norm(vect1)
	# dist2 = np.linalg.norm(vect2)
	# if dist1 > dist2:
	# 	dist1, dist2 = dist2, dist1
	# 	vect1, vect2 = vect2, vect1
	# 	x4, x2 = x2, x4
	# 	y4, y2 = y2, y4

	# smallVect1 = vect1 / 5

	# rectangle = [ (x1 + smallVect1[0]*2, y1 + smallVect1[1]*2), ( x1 + smallVect1[0]*3, y1 + smallVect1[1]*3) ]
	# rectangle += [ (x4 + smallVect1[0]*3, y4 + smallVect1[1]*3), (x4 + smallVect1[0]*2, y4 + smallVect1[1]*2 ) ]


	# ----------- maxInscribedcircle() end ------------------------------

	# print(maxCircle)
	# img = np.zeros( (frameShape[0], frameShape[1], 3))
	# cv2.drawContours(img,[contour],0,(255,255,255),2)
	# cv2.circle(img,center, radius, (0,255,0), 0)
	# cv2.circle(img, center, 5, (0,255,255), -1  )
	# cv2.drawContours(img,[ np.array(rectangle, dtype=np.int0)],0,(255,0,0),2)
	# cv2.imshow('img', img)
	# cv2.waitKey(0)