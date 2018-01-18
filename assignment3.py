import cv2
import numpy as np
from scipy import misc

# descriptor size
n = 11

# load images
# template = cv2.imread("tsukuba/scene1.row3.col1.ppm", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("tsukuba/scene1.row3.col2.ppm", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("venus/im0.ppm", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("venus/im6.ppm", cv2.IMREAD_GRAYSCALE)
# template = cv2.imread("map/im0.pgm", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("map/im1.pgm", cv2.IMREAD_GRAYSCALE)

def getDescriptors(img, y, n):
    descriptors = []
    for x in range(n//2, len(img[y])-n//2):
        try:
            nd = []
            for i in range(int(x)-(n//2), int(x)+(n//2)):
                for j in range(int(y)-(n//2), int(y)+(n//2)):
                    nd.append(img[j,i])
            descriptors.append(nd)
        except IndexError :
            print("index error")
    return descriptors

# create pyramid
def buildPyramid(img):
    pyramide = []
    pyramide.append(img)
    for i in range(1,4):
        low_res = cv2.pyrDown(pyramide[i-1])
        pyramide.append(low_res)
    return pyramide

def getLocalMean(x, y, scale):
    dis = []
    for i in range(x-n//2, x+n//2):
        for j in range(y-n//2, y+n//2):
            row = int(i/scale)
            col = int(j/scale)
            dis.append(disparities[row,col])
    return np.mean(dis)

# first imagae
p1 = buildPyramid(img)
p2 = buildPyramid(template)
height, width = p1[3].shape
disparities = np.zeros((height, width))
depth = np.zeros((height, width))
for l in range(0, height-n//2):
    des = getDescriptors(p1[3], l, n)
    des2 = getDescriptors(p2[3], l, n)
    d1 = np.asarray(des)
    d2 = np.asarray(des2)
    column = n//2
    for index, descriptor in enumerate(d1):
        highest_corr = -1
        for index2, descriptor2 in enumerate(d2):
            result = cv2.matchTemplate(descriptor, descriptor2, cv2.TM_CCORR_NORMED)

            if(result > highest_corr):
                highest_corr = result
                disparities[l, column] = (index2 - index)
                depth[l, column] = (index2 - index) *  (1.0 / 16.0)
        if(depth[l, column] == 0.0):
            disparities[l, column] = getLocalMean(l, column, 1)
            depth[l, column] = disparities[l,column] * (1.0 / 16.0)
        column += 1

# second image
height, width = p1[2].shape
depth = np.zeros((height, width))
for l in range(0, height-n//2):
    des = getDescriptors(p1[2], l, n)
    des2 = getDescriptors(p2[2], l, n)
    d1 = np.asarray(des)
    d2 = np.asarray(des2)
    column = n//2
    for index, descriptor in enumerate(d1):
        highest_corr = -1
        int_disparities = disparities.astype(int)
        row = int(l/2)
        col = int(column/2)
        for k in range(index - int_disparities[row,col] * 2, index + int_disparities[row,col] * 2):
            i = k
            if(i<0):
                i = 0
            elif(i > width - n):
                i = width - n
            descriptor2 = d2[i]
            result = cv2.matchTemplate(descriptor, descriptor2, cv2.TM_CCORR_NORMED)
            if(result > highest_corr):
                highest_corr = result
                disparities[row, col] = (k - index)
                depth[l, column] = (k - index) *  (1.0 / 16.0)
        if(depth[l, column] == 0.0):
            disparities[row, col] = getLocalMean(l, column, 2)
            depth[l, column] = disparities[row, col] * (1.0 / 16.0)
        column += 1

# third image
height, width = p1[1].shape
depth = np.zeros((height, width))
for l in range(0, height-n//2):
    des = getDescriptors(p1[1], l, n)
    des2 = getDescriptors(p2[1], l, n)
    d1 = np.asarray(des)
    d2 = np.asarray(des2)
    column = n//2
    for index, descriptor in enumerate(d1):
        highest_corr = -1
        int_disparities = disparities.astype(int)
        row = int(l/4)
        col = int(column/4)
        for k in range(index - int_disparities[row,col] * 4, index + int_disparities[row,col] * 4):
            i = k
            if(i<0):
                i = 0
            elif(i > width - n):
                i =  width - n
            descriptor2 = d2[i]
            result = cv2.matchTemplate(descriptor, descriptor2, cv2.TM_CCORR_NORMED)
            if(result > highest_corr):
                highest_corr = result
                disparities[row, col] = (k - index)
                depth[l, column] = (k - index) *  (1.0 / 16.0)
        if(depth[l, column] == 0.0):
            disparities[row, col] = getLocalMean(l, column, 4)
            depth[l, column] = disparities[row, col] * (1.0 / 16.0)
        column += 1

# final image
height, width = p1[0].shape
depth = np.zeros((height, width))
for l in range(0, height-n//2):
    des = getDescriptors(p1[0], l, n)
    des2 = getDescriptors(p2[0], l, n)
    d1 = np.asarray(des)
    d2 = np.asarray(des2)
    column = n//2
    for index, descriptor in enumerate(d1):
        highest_corr = -1
        int_disparities = disparities.astype(int)
        row = int(l/8)
        col = int(column/8)
        for k in range(index - int_disparities[row,col] * 8, index + int_disparities[row,col] * 8):
            i = k
            if(i<0):
                i = 0
            elif(i > width - n):
                i = width - n
            descriptor2 = d2[i]
            result = cv2.matchTemplate(descriptor, descriptor2, cv2.TM_CCORR_NORMED)

            if(result > highest_corr):
                highest_corr = result
                disparities[row, col] = (k - index)
                depth[l, column] = (k - index) *  (1.0 / 16.0)
        # ensure smoothness
        if(depth[l, column] == 0.0):
            disparities[row, col] = getLocalMean(l, column, 8)
            depth[l, column] = disparities[row, col] * (1.0 / 16.0)
        column += 1

cv2.imshow('test', depth)
# pic = depth * 255
# pic = pic.astype('uint8')
# cv2.imwrite('test-venus11-18.png', pic)

cv2.waitKey(0)
cv2.destroyAllWindows()
