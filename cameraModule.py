import cv2
import numpy as np


class GlobalParams:
	yRange = (54, 163)
	crRange = (133, 173)
	cbRange = (77, 127)
	backgroundFilePath = 'background.png'

def runCamera():

    cap = cv2.VideoCapture(0)

    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()

       	normalizedFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

       	background = backgroundSubstraction(normalizedFrame, GlobalParams.backgroundFilePath)

       	binaryForeground = processChannels(background)
       	foreground = cv2.bitwise_and(normalizedFrame,normalizedFrame, mask = binaryForeground)

       	faces = faceRectangles(frame)
       	foreground = faceRemoval(foreground, faces)

       	handsBinary = getSkin(foreground, GlobalParams.yRange, GlobalParams.crRange, GlobalParams.cbRange)

       	out = cv2.cvtColor(foreground, cv2.COLOR_YCrCb2BGR)
        cv2.imshow('frame',handsBinary)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    # out.release()
    cv2.destroyAllWindows()



def backgroundSubstraction(image, backgroundFilePath):
	
	background = cv2.imread(backgroundFilePath)
	background = cv2.cvtColor(background, cv2.COLOR_BGR2YCrCb)
	result = np.absolute( np.subtract(image, background) )

	return result



def faceRectangles(rgbImage):

	gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier('E:\Programe\openCV\opencv\data\haarcascades\haarcascade_frontalface_default.xml')
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	return faces



def processChannels(normalizedImage):

	channelY = processChannel(normalizedImage[:,:,0], 50, 255 )
	channelCr = processChannel(normalizedImage[:,:,1], 50, 255 )
	channelCb = processChannel(normalizedImage[:,:,2], 50, 255 )
	cv2.imshow('binaryY', channelY)
	cv2.imshow('binaryCr', channelCr)
	cv2.imshow('binaryCb', channelCb)
	# binary = cv2.add(channelY, channelCr)
	# binary = cv2.add(binary, channelCb)
	binary= cv2.bitwise_and(channelY,channelCb)
	binary= cv2.bitwise_and(binary,channelCr)
	kernel = np.ones((5, 5), np.uint8)
	mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
	cv2.imshow('mask', mask)
	return binary



def processChannel(channel, lowerValue, upperValue):
	
	mask = cv2.inRange(channel, lowerValue, upperValue)
	# morphological transformations
	kernel = np.ones((5, 5), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	return mask



def faceRemoval(image, faces):
	for (x,y,w,h) in faces:
		image[ y:y+int(h*1.5), x:x+w, : ] = 0

	return image



def cannyEdge():

	return 0



def getSkin(normalized, yRange, crRange, cbRange ):
	lower_skin = np.array([yRange[0], crRange[0], cbRange[0] ])
	upper_skin = np.array([yRange[1], crRange[1], cbRange[1] ])

	mask = cv2.inRange(normalized, lower_skin, upper_skin)

	# morphological transformations
	kernel = np.ones((5, 5), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	normalized = cv2.GaussianBlur(normalized, (5, 5), 1, 1)

	return mask


runCamera()