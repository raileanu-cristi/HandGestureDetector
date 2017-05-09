import numpy as np
import cv2
import math


class Detector:
    SAMPLECOLOR = 0
    DETECTION = 1

    def __init__(self):
        self.state = self.SAMPLECOLOR
        self.rectanglePos = []
        self.radius = 10
        self.binHand = False

    #
    # Reset the detector
    #
    def reset(self):
        self.state = self.SAMPLECOLOR
        self.binHand = False

    #
    # Makes the settings
    #
    def set(self, rectangles, _dim, _color):
        self.rectanglePos = rectangles
        self.rectangleDim = _dim
        self.rectangleColor = _color

    #
    #   sampleColors
    #
    def sampleColors(self, img):

        if (self.state == self.DETECTION):
            return

        self.colors = []

        nrColors = len(self.rectanglePos)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for i in range(0, len(self.rectanglePos)):
            x = self.rectanglePos[i][0]
            y = self.rectanglePos[i][1]

            self.colors.append(hsv[x, y])

        # color sampling complete, change machine state
        self.state = self.DETECTION


    #
    #
    #
    def __drawRectangle(self, img, x, y):

        d = self.rectangleDim
        color = self.rectangleColor
        lineWidth = 1
        cv2.rectangle(img, (x - d, y - d), (x + d, y + d), color, lineWidth)
        return


        
    #   
    #   detects the skin from the HSV image
    #
    def getSkin(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # define range of skin color in HSV
        lower_skin = np.array([5, 38, 51])
        upper_skin = np.array([17, 250, 242])

        hsv = cv2.GaussianBlur(hsv, (7, 7), 1, 1)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        mask = cv2.medianBlur(mask, ksize=7)
        mask = cv2.medianBlur(mask, ksize=7)

        # cv2.imshow('binHand', mask) # debug use ------------

        # morphological transformations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # mask = cv2.dilate(mask,kernel,iterations = 1)
        # cv2.imshow('Binary', mask)  # debug use ------------

        # mask should be RGB
        return mask


    def getSkin2(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # define range of skin color in HSV
        lower_skin = np.array([5, 38, 31])
        upper_skin = np.array([17, 250, 242])

        hsv = cv2.GaussianBlur(hsv, (5, 5), 1, 1)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # mask = cv2.medianBlur(mask, ksize=5)
        # mask = cv2.medianBlur(mask, ksize=7)

        # cv2.imshow('binHand', mask) # debug use ------------

        # morphological transformations
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        mask = cv2.dilate(mask,kernel,iterations = 1)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 1)
        # cv2.imshow('Binary', mask)  # debug use ------------

        # mask should be RGB
        return mask


    #
    #
    #
    def getConvexPointsInCoutour(self, img, binImage):
        # print(np.median(binImage))
        _, contours, hierarchy = cv2.findContours(binImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        
        # if np.array_equal(binImage, binImage2):
        # 	print("Equal!")
        # cv2.imshow('Binary with countour', binImage2)


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
        cv2.drawContours(img, [hull], contourIdx=0, color=(255, 0, 0), thickness=1)

        # center of the contour area
        moments = cv2.moments(cnt)
        if moments['m00']!=0:
            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
            cy = int(moments['m01']/moments['m00']) # cy = M01/M00

   
        centr=(cx,cy)       
        cv2.circle(img,centr,5,[0,0,255],2)       


        # convexity defects
        #
        hull = cv2.convexHull(cnt, returnPoints = False)    
        defects = cv2.convexityDefects(cnt,hull)

        if defects==None:
            return img

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
            print( teta )
            if teta > 80:
                continue

            cv2.circle(img,start,4,[255,255,0],-1)
            # end
            # cv2.circle(img,end,2,[0,255,255],-1)
            # far
            cv2.circle(img,far,4,[0,0,255],-1)

            # dist = cv2.pointPolygonTest(cnt,centr,True)
        #     cv2.line(img,start,end,[0,255,0],2)                
        #     cv2.circle(img,far,5,[0,0,255],-1)
        # exit() # debug
        # print(defects)
        return img

    


    #
    # Returns the image with canvas
    #
    def getCanvas(self, img):

        if (self.state == self.SAMPLECOLOR):

            # draw rectangles
            if (len(self.rectanglePos) == 0):
                print("Rectangles not set")
                return img

            for i in range(0, len(self.rectanglePos)):
                self.__drawRectangle(img, self.rectanglePos[i][0], self.rectanglePos[i][1])

            return img
        else:
            
            # get the binary representation of the hand
            binHand = self.getSkin2(img)

            cv2.imshow('Binary', binHand)  # debug use ------------

            img = self.getConvexPointsInCoutour(img, binHand)

            if (not self.binHand):
                cv2.imwrite('binHand.jpg', binHand)
                # cv2.imshow('binHand', binHand)

                self.binHand = True

            rez = img

            output = rez

        return output


#
# ---------------------------------------------- DetectorSingleton ---------------------------------------------
#
class DetectorSingleton:
    __INSTANCE = Detector()

    def getInstance():
        return DetectorSingleton.__INSTANCE