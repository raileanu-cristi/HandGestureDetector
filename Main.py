import numpy as np
import cv2
from appJar import gui
from Detector import DetectorSingleton

# http://sa-cybernetics.github.io/blog/2013/08/12/hand-tracking-and-recognition-with-opencv/
# https://www.intorobotics.com/9-opencv-tutorials-hand-gesture-detection-recognition/
# http://appjar.info/pythonWidgets/#
# shape descriptor
# shape context descriptor
# invarianta la rotatie, scalare, translatie
# nearest neighbour, svm, neural network
# medial axis - scheleton of the object
# discriminare intre 5,4,..1 degete ridicate
# discriminare intre 2 semne cu acelasi nr de semne, dar degete diferite


def runCamera():

    e0 = cv2.getTickCount()

    cap = cv2.VideoCapture(0)

    sampleTime = 0 # seconds

    frameSkipRatio = 10
    frameNr = 0

    while(True):

        frameNr += 1
        frameNr %= 10000

        # Capture frame-by-frame
        ret, frame = cap.read()

        if frameNr % frameSkipRatio != 0:
            continue

        #img = cv2.Canny(frame, 100, 200) # canny edge detection
        e = cv2.getTickCount()
        t = (e - e0) / cv2.getTickFrequency()

        if t >= sampleTime:
            DetectorSingleton.getInstance().sampleColors(frame)

        canvas = DetectorSingleton.getInstance().getCanvas(frame)
        # Display the resulting frame
        cv2.imshow('frame',canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


    # def menuPress():
    #     a = 0

def SettingsWindow(btn):

    window = gui()
    # initial settings
    window.setTitle('Settings')
    window.setGeometry("600x500")

    # widgets
    window.setSticky("sw")
    row = window.getRow()
    window.addButton("OK", window.stop, row,0)
    window.addButton("Cancel", window.stop, row,1)
    window.setButtonPadding("OK", [10,10])

    window.go()


#
#
#
def startVideoClick(btn):
    DetectorSingleton.getInstance().reset()
    runCamera()


#
#
#
def myCustumGUI():

    app = gui()
    # initial settings
    app.setTitle('Hand Gesture Recognition 1.01')
    app.setGeometry("600x500")

    # window dropdown menu
    fileMenuStrings = ["Close"]
    fileMenuFunctions = [app.stop]
    app.addMenuList('File', fileMenuStrings, fileMenuFunctions)

    toolsMenuStrings = ["Settings"]
    toolsMenuFunctions = [SettingsWindow]
    app.addMenuList('Tools', toolsMenuStrings, toolsMenuFunctions)

    aboutMenuStrings = ["About"]
    aboutMenuFunctions = [None]
    app.addMenuList('Help', aboutMenuStrings, aboutMenuFunctions)

    # main window widgets
    app.addButton("Start video", startVideoClick)

    return app



def main():
    
    app = myCustumGUI()

    #detector settings
    x = 100
    y = 350
    rectangleCenters = [[x,y-20], [x,y+20], [x,y+50], [x + 40, y + 40]]
    dim = 10
    rColor = (0,100,0)
    DetectorSingleton.getInstance().set(rectangleCenters,dim,rColor)

    #starting app
    app.go()

# --------------------------------------------------------------------------------------------------------
# ------------------------------------------------ Main --------------------------------------------------
# --------------------------------------------------------------------------------------------------------

main()
