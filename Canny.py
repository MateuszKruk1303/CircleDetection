import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import convolve
from matplotlib import pyplot as plt
from PIL import Image

img = Image.open("Resources/120x2O100x2O50x2O20x4.png").convert("L")
img_Gray = np.asarray(img, dtype=np.uint8)
imgGray = img_Gray.astype(np.uint8)
cv2.imshow("Grayscale IMG", imgGray)

# Filtr Gaussa
def gauss_filter(img, size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1] # Przygotowanie tablic dla maski
    denom = 1 / (2.0 * np.pi * sigma**2) # Mianownik rownania maski
    mask = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * denom # Rownanie na maske filtru
    I = convolve(img, mask) # Splot obrazu oraz maski
    return I

# Wyznaczanie gradientu
def prewitt_operator(img):
    Ixx = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]]) # Pierwsza pochodna, kierunek poziomy

    Iyy = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]]) # Pierwsza pochodna, kierunek pionowy

    Ix = ndimage.filters.convolve(img, Ixx) # Splot pochodnej i obrazu
    Iy = ndimage.filters.convolve(img, Iyy) # Splot pochodnej i obrazu

    I = np.sqrt(np.square(Ix) + np.square(Iy)) # Rownanie obrazu wynikowego
    I = I / I.max() * 255
    theta = np.arctan2(Iy, Ix) # Obliczenie katu detekcji krawedzi
    return (I, theta)

# Tłumienie niemaksymalne
def suppres(img, theta):
    x, y = img.shape
    I = np.zeros((x, y), dtype=np.uint8)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1, x-1):
        for j in range (1, y-1):
            try:
                q = 255
                r = 255
                if(0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180): # kat 0
                    q = img[i, j+1]
                    r = img[i, j-1]
                elif (22.5 <= angle[i, j] < 67.5): # kat 45
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                elif (67.5 <= angle[i, j] < 112.5): # kat 90
                    q = img[i+1, j]
                    r = img[i-1, j]
                elif (112.5 <= angle[i, j] < 157.5): # kat 135
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]
                if (img[i, j] >= q) and (img[i, j] >= r):
                    I[i, j] = img[i, j]
                else:
                    I[i, j] = 0
            except IndexError as e:
                pass
    return I

# Podwojne progowanie
def doubleThresh(img, lThreshVal=0.06, hThreshVal=0.1):
    hThresh = img.max() * hThreshVal
    lThresh = hThresh * lThreshVal
    x, y = img.shape
    I = np.zeros((x, y), dtype=np.uint8)
    weak = np.uint8(30)
    strong = np.uint8(255)
    strong_i, strong_j = np.where(img >= hThresh)
    weak_i, weak_j = np.where((img <= hThresh) & (img >= lThresh))
    I[strong_i, strong_j] = strong
    I[weak_i, weak_j] = weak
    return (I, weak, strong)

# Detekcja krawedzi histereza
def hysteresis(img, weak, strong=255):
    x, y = img.shape
    for i in range(1, x - 1):
        for j in range(1, y - 1):
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

def detectCircles(img, radius):
    edges = img
    plt.imshow(img)
    plt.imshow(edges, cmap='gray')
    plt.show()
    rows, columns = edges.shape
    img2 = edges

    radius += 2

    for x in range(0, columns):
        for y in range(0, rows):
            if (edges[y, x] == 255):
                for ang in range(0, 360):
                    t = (ang * np.pi) / 180
                    x0 = int(round(x - radius * np.cos(t)))
                    y0 = int(round(y - radius * np.sin(t)))
                    if (x0 < columns and x0 > 0 and y0 < rows and y0 > 0 and img2[y0, x0] < 255):
                        img2[y0, x0] += 1

    maxes = np.argwhere((img2 > 200) & (img2 < 250)).flatten()

    for i in range(0, len(maxes), 2):
        cv2.circle(img2, center=(maxes[i + 1], maxes[i]), radius=radius, color=(255, 255, 255), thickness=2)

    plt.imshow(img2, cmap='gray')
    plt.show()

x, y = imgGray.shape
img_gauss = np.zeros((x, y), dtype=np.uint8)
img_gauss = gauss_filter(imgGray, 5, 1.4)
cv2.imshow("Smoothing image - Gaussian filter", img_gauss)
cv2.waitKey(0)

img_gauss = np.asarray(img_gauss, dtype=np.int32)
img_prewitt = np.zeros((x, y), dtype=np.int32)
img_prewitt, theta = prewitt_operator(img_gauss)
img_prewitt = np.array(img_prewitt)
img_prewitt = img_prewitt.astype(np.uint8)
cv2.imshow("Prewitt operator - finding the intensity of the image", img_prewitt)
cv2.waitKey(0)

img_supp = np.zeros((x, y), dtype=np.uint8)
img_supp = suppres(img_prewitt, theta)
img_supp = np.array(img_supp)
img_supp = img_supp.astype(np.uint8)
cv2.imshow("Non-Maximum Suppression", img_supp)
cv2.waitKey(0)

img_threshold = np.zeros((x, y), dtype=np.uint8)
img_threshold, weak, strong = doubleThresh(img_supp)
img_threshold = np.array(img_threshold)
img_threshold = img_threshold.astype(np.uint8)
cv2.imshow("Double Threshold", img_threshold)
cv2.waitKey(0)

img_edge = np.zeros((x, y), dtype=np.uint8)
img_edge = hysteresis(img_threshold, weak, strong)
cv2.imshow("Edge tracked by hysteresis", img_edge)
cv2.waitKey(0)