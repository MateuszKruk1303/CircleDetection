import numpy as np
import cv2
from matplotlib import pyplot as plt
import imageio

def edgeDetectionTemplate(img):
    edges = cv2.Canny(img, 100, 200)
    return edges


def detectCircles(img, radius):
    edges = edgeDetectionTemplate(img)
    plt.imshow(img)
    plt.imshow(edges, cmap='gray')
    plt.show()
    rows, columns = edges.shape
    img2buffer = np.zeros([rows, columns], dtype=np.uint8)

    plt.imshow(img2buffer, cmap='gray')
    plt.show()


    radius += 2

    for x in range(0, columns):
        for y in range(0, rows):
            if (edges[y, x] == 255):
                for ang in range(0, 360):
                    t = (ang * np.pi) / 180
                    x0 = int(round(x - radius * np.cos(t)))
                    y0 = int(round(y - radius * np.sin(t)))
                    if (x0 < columns and x0 > 0 and y0 < rows and y0 > 0 and edges[y0, x0] < 255):
                        img2buffer[y0, x0] += 1

    maxes = np.argwhere((img2buffer > 200) & (img2buffer < 250)).flatten()

    plt.imshow(img2buffer, cmap='gray')
    plt.show()

    for i in range(0, len(maxes), 2):
        cv2.circle(edges, center=(maxes[i + 1], maxes[i]), radius=radius, color=(255, 255, 255), thickness=2)

    plt.imshow(edges)
    plt.show()


img = imageio.imread('circle50x50.png')

detectCircles(img, 23)




