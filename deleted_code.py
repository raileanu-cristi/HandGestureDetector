
def __getBinaryHand(self,image):

    colors = self.colors
    radius = self.radius
    h,w,c = image.shape;

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Gaussian Blur
    hsv = cv2.GaussianBlur(hsv, (7,7), 1,1)

    nrColors = len(colors);
    threshholds = np.zeros((nrColors,h,w) );

    #important
    delta = [20, 100, 200]

    for i in range(0, nrColors):
        mask = cv2.inRange(hsv, colors[i] - delta, colors[i] + delta )
        threshholds[i] = mask
    # print(radius)
    # print(colors[0])
    # print(colors[0] - delta)
    # print(threshholds[0])
    output = np.sum(threshholds, axis=0)
    #print(output)
    #print(output)
    return output