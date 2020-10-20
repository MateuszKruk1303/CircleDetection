import numpy as np
import cv2
from matplotlib import pyplot as plt


def edgeDetectionTemplate():
    img = cv2.imread('monety2.png', 0)
    edges = cv2.Canny(img, 100, 200)

    plt.imshow(edges, cmap='gray')
    plt.title('Template')

    plt.show()


edgeDetectionTemplate()
