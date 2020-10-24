import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import convolve
from matplotlib import pyplot
from matplotlib import image
from PIL import Image

img = Image.open("Resources/4.jpg").convert("L")
img_Gray = np.asarray(img, dtype="int32")
imgGray = img_Gray.astype(np.uint8)
cv2.imshow("Grayscale IMG", imgGray)

# Gaussian Blur 5x5 kernel
def gaussian_kernel(size, sigma = 1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filter(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    I = np.hypot(Ix, Iy)
    #I = np.sqrt(np.square(Ix) + np.square(Iy))
    I = I / I.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (I, theta)

def no_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range (1, N-1):
            try:
                q = 255
                r = 255

                #angle 0
                if(0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i,j ] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass
    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i, j] == weak):
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    return img

x, y = imgGray.shape
result = gaussian_kernel(5, 1.2)
img_gauss = np.zeros((x, y), dtype=np.uint8)
img_gauss = convolve(imgGray, result)
cv2.imshow("Blur", img_gauss)
cv2.waitKey(0)

img_gauss = np.asarray(img_gauss, dtype=np.int32)
img_sobel = np.zeros((x, y), dtype=np.int32)
img_sobel, theta = sobel_filter(img_gauss)
img_sobel = np.array(img_sobel)
img_sobel = img_sobel.astype(np.uint8)
cv2.imshow("Sobel Filter", img_sobel)
cv2.waitKey(0)

img_surp = np.zeros((x, y), dtype=np.uint8)
img_surp = no_max_suppression(img_sobel, theta)
img_surp = np.array(img_surp)
img_surp = img_surp.astype(np.uint8)
cv2.imshow("Max Surpression", img_surp)
cv2.waitKey(0)

img_threshold = np.zeros((x, y), dtype=np.int32)
img_threshold, weak, strong = threshold(img_surp)
img_threshold = np.array(img_surp)
cv2.imshow("Threshold", img_threshold)
cv2.waitKey(0)

img_edge = np.zeros((x, y), dtype=np.uint8)
img_edge = hysteresis(img_surp, weak, strong)
cv2.imshow("Edge tracked", img_edge)
cv2.waitKey(0)