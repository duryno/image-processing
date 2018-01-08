import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage

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

def getLocalMean(x, y, width, height, scale):
    print(x)
    print(width)

    dis = []
    if((int(x)-(n//2)) >= 0):
        safeMinX = int(x)-(n//2)
    else:
        safeMinX = 0

    if((int(x)+(n//2)+1) < width):
        safeMaxX = int(x)+(n//2)+1
    else:
        safeMaxX = width

    if((int(y)-(n//2)) >= 0):
        safeMinY = int(y)-(n//2)
    else:
        safeMinY = 0

    if((int(y)+(n//2)+1) < height):
        safeMaxY = int(y)+(n//2)+1
    else:
        safeMaxY = height

    for i in range(int(safeMinX), int(safeMaxX)):
        for j in range(int(safeMinY), int(safeMaxY)):
            row = int(i/scale)
            col = int(j/scale)
            dis.append(disparities[row,col])
    return np.mean(dis)

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
        if(depth[l, column] == 0.0):
            disparities[l, column] = getLocalMean(l, column, width, height, 1)
            depth[l, column] = 2 * (1.0 / 16.0)
        correlations[l, column] = min(row_corrs)
        column += 1

cv2.imshow('test3', depth)
cv2.imwrite('test3.png', depth)
misc.imsave('test31.jpg', depth)


height, width = p1[2].shape
correlations = np.zeros((height, width))
depth = np.zeros((height, width))
for l in range(n, height-n):
    des = getDescriptors(p1[2], l, n)
    des2 = getDescriptors(p2[2], l, n)
    d1 = np.asarray(des)
    d2 = np.asarray(des2)
    column = n
    for index, descriptor in enumerate(d1):
        row_corrs = []
        smallest_corr = 1
        int_disparities = disparities.astype(int)
        row = int(l/2)
        col = int(column/2)
        for k in range(index - int_disparities[row,col] * 2, index + int_disparities[row,col] * 2):
            i = k
            if(i<0):
                i = 0
            elif(i > width - 2*n - 1):
                i = width - 2*n - 1
            descriptor2 = d2[i]
            index2 = k
            result = cv2.matchTemplate(descriptor, descriptor2, cv2.TM_SQDIFF_NORMED)
            row_corrs.append(result)
            if(result < smallest_corr):
                smallest_corr = result
                disparities[row, col] = (index2 - index)
                depth[l, column] = (index2 - index) *  (1.0 / 16.0)
        if(depth[l, column] == 0.0):
            disparities[row, col] = getLocalMean(l, column, width, height, 2)
            depth[l, column] = 2 * (1.0 / 16.0)
        column += 1

cv2.imshow('test2', depth)

height, width = p1[1].shape
correlations = np.zeros((height, width))
depth = np.zeros((height, width))
for l in range(n, height-n):
    des = getDescriptors(p1[1], l, n)
    des2 = getDescriptors(p2[1], l, n)
    d1 = np.asarray(des)
    d2 = np.asarray(des2)
    column = n
    for index, descriptor in enumerate(d1):
        row_corrs = []
        smallest_corr = 1
        int_disparities = disparities.astype(int)
        row = int(l/4)
        col = int(column/4)
        for k in range(index - int_disparities[row,col] * 4, index + int_disparities[row,col] * 4):
            i = k
            if(i<0):
                i = 0
            elif(i > width - 2*n - 1):
                i = width - 2*n - 1
            descriptor2 = d2[i]
            index2 = k
            result = cv2.matchTemplate(descriptor, descriptor2, cv2.TM_SQDIFF_NORMED)
            row_corrs.append(result)
            if(result < smallest_corr):
                smallest_corr = result
                disparities[row, col] = (index2 - index)
                depth[l, column] = (index2 - index) *  (1.0 / 16.0)
        if(depth[l, column] == 0.0):
            disparities[row, col] = getLocalMean(l, column, width, height, 4)
            depth[l, column] = 2 * (1.0 / 16.0)
        column += 1

cv2.imshow('test1', depth)

height, width = p1[0].shape
correlations = np.zeros((height, width))
depth = np.zeros((height, width))
for l in range(n, height-n):
    des = getDescriptors(p1[0], l, n)
    des2 = getDescriptors(p2[0], l, n)
    d1 = np.asarray(des)
    d2 = np.asarray(des2)
    column = n
    for index, descriptor in enumerate(d1):
        row_corrs = []
        smallest_corr = 1
        int_disparities = disparities.astype(int)
        row = int(l/8)
        col = int(column/8)
        for k in range(index - int_disparities[row,col] * 8, index + int_disparities[row,col] * 8):
            i = k
            if(i<0):
                i = 0
            elif(i > width - 2*n - 1):
                i = width - 2*n - 1
            descriptor2 = d2[i]
            index2 = k
            result = cv2.matchTemplate(descriptor, descriptor2, cv2.TM_SQDIFF_NORMED)
            row_corrs.append(result)

            if(result < smallest_corr):
                smallest_corr = result
                disparities[row, col] = (index2 - index)
                depth[l, column] = (index2 - index) *  (1.0 / 16.0)
        if(depth[l, column] == 0.0):
            disparities[row, col] = getLocalMean(l, column, width, height, 8)
            depth[l, column] = 2 * (1.0 / 16.0)
        column += 1

cv2.imshow('test', depth)

cv2.waitKey(0)
cv2.destroyAllWindows()
