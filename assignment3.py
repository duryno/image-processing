import cv2
import numpy as np
from matplotlib import pyplot as plt

# descriptor size
n = 5

# load images
template = cv2.imread("tsukuba/scene1.row3.col1.ppm", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("tsukuba/scene1.row3.col2.ppm", cv2.IMREAD_GRAYSCALE)

# convert images to float
template= np.float32(template)
img = np.float32(img)

def getDescriptor(img, x, y, n):
    descriptor = []
    try:
        for i in range(int(x)-(n//2), int(x)+(n//2)+1):
            for j in range(int(y)-(n//2), int(y)+(n//2)+1):
                descriptor.append(img[j,i])
    except IndexError :
        print("index error")
    return descriptor

# the following functions can be used for cross-correlation
# cv2.MatchTemplate
# scipy.signal.correlate

cv2.waitKey(0)
cv2.destroyAllWindows()
