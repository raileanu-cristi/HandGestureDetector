import cv2
import numpy as np
from HandFeaturesDetection import getConvexPointsInCoutour



def getSkin(image):

	B, G, R = image[:,:,0], image[:,:,1], image[:,:,2]
	rgbSum = np.sum(image, axis=2)
	rgbSum += np.ones(rgbSum.shape, dtype=np.uint8)

	r = R / rgbSum
	g = G / rgbSum
	h, w = R.shape
	rgImage = np.zeros( (h,w,2) )
	rgImage[:,:,0], rgImage[:,:,1] = r, g
	# print(np.mean(r))
	# print(r[10,10], g[10,10], R[10,10], G[10,10], B[10,10], rgbSum[10,10] )
	# lower_skin = np.array([0.20, 0.25])
	# upper_skin = np.array([0.25, 0.40])
	lower_skin = np.array([0.40, 0.25])
	upper_skin = np.array([0.65, 0.35])
	# print(lower_skin.dtype)
	rgImage = cv2.GaussianBlur(rgImage, (5, 5), 1, 1)
	
	mask = cv2.inRange(rgImage, lower_skin, upper_skin)
	
	# ret, mask = cv2.threshold(mask, 140, 255, cv2.THRESH_BINARY)
	# morphological transformations
        
	kernel = np.ones((5, 5), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	
	# mask = cv2.dilate(mask,kernel,iterations = 1)
	kernel = np.ones((5, 5), np.uint8)
	# mask = cv2.dilate(mask,kernel,iterations = 1)

	# print('mask shape ' + str(mask.shape))
	return mask


def getSkin2(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # define range of skin color in HSV
        lower_skin = np.array([5, 38, 31])
        upper_skin = np.array([17, 250, 242])

        hsv = cv2.GaussianBlur(hsv, (5, 5), 1, 1)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # morphological transformations
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        mask = cv2.dilate(mask,kernel,iterations = 1)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 1)

        # print('mask shape ' + str(mask.shape))
        # mask should be RGB
        return mask


def getSkin3(image):
        normalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # define range of skin color in HSV
        lower_skin = np.array([30, 133, 77]) # 77≤Cb≤127 and 133≤Cr≤173
        upper_skin = np.array([255, 173, 127]) 

        normalized = cv2.GaussianBlur(normalized, (5, 5), 1, 1)

        mask = cv2.inRange(normalized, lower_skin, upper_skin)

        # morphological transformations
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # print('mask shape ' + str(mask.shape))
        # mask should be RGB
        return mask
