import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig

# descriptor size
n = 5

# load images
template = cv2.imread("tsukuba/scene1.row3.col1.ppm", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("tsukuba/scene1.row3.col2.ppm", cv2.IMREAD_GRAYSCALE)

def getDescriptors(img, y, n):
    descriptors = []
    for x in range(n, len(img[y])-n):
        try:
            nd = []
            # x = round(k.pt[0])
            # y = round(k.pt[1])
            for i in range(int(x)-(n//2), int(x)+(n//2)+1):
                for j in range(int(y)-(n//2), int(y)+(n//2)+1):
                    nd.append(img[j,i])
            descriptors.append(nd)
            # print(descriptors)
        except IndexError :
            print(str(x))
            print("index error")
    return descriptors

# create pyramide
def buildPyramid(img):
    pyramide = []
    pyramide.append(img)
    for i in range(0,3):
        low_res = cv2.pyrDown(pyramide[i-1])
        pyramide.append(low_res)
    return pyramide

p1 = buildPyramid(img)
p2 = buildPyramid(template)
height, width = p1[3].shape
correlations = np.zeros((height, width))
disparities = np.zeros((height, width))
depth = np.zeros((height, width))
for l in range(n, height-n):
    des = getDescriptors(p1[3], l, n)
    des2 = getDescriptors(p2[3], l, n)
    d1 = np.asarray(des)
    d2 = np.asarray(des2)
    column = n
    for index, descriptor in enumerate(d1):
        disparities[l, column] = width
        row_corrs = []
        smallest_corr = 1
        for index2, descriptor2 in enumerate(d2):
            result = cv2.matchTemplate(descriptor, descriptor2, cv2.TM_SQDIFF_NORMED)
            row_corrs.append(result)
            if(result < smallest_corr):
                smallest_corr = result
                disparities[l, column] = (index2 - index)
                depth[l, column] = (index2 - index) *  (1.0 / 16.0)
        # print(min(row_corrs))
        correlations[l, column] = min(row_corrs)
        column += 1

cv2.imshow('test', depth)

cv2.waitKey(0)
cv2.destroyAllWindows()
