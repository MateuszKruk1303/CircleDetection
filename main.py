import numpy as np
import cv2
from matplotlib import pyplot as plt
import imageio

def edgeDetectionTemplate():
    img = cv2.imread('monety2.png', 0)
    edges = cv2.Canny(img, 100, 200)

    plt.imshow(edges, cmap='gray')
    plt.title('Template')

    plt.show()


img = imageio.imread('120x2O100x2O50x2O20x4.png')
edges = cv2.Canny(img, 100, 200)
plt.imshow(img)
plt.imshow(edges, cmap='gray')
plt.show()
rows, columns = edges.shape
img2 = edges

radius = 10+2


for x in range(0, columns):
    for y in range(0, rows):
        if(edges[y, x] == 255):
            for ang in range(0, 360):
                t = (ang*np.pi)/180
                x0 = int(round(x - radius * np.cos(t)))
                y0 = int(round(y - radius * np.sin(t)))
                if(x0<columns and x0>0 and y0<rows and y0>0):
                    img2[y0, x0] += 1


maxes = np.argwhere((img2>105)&(img2<250)).flatten()

print(int(len(maxes)/2))

for i in range(0, len(maxes), 2):
    print(i)
    cv2.circle(img2, center=(maxes[i+1], maxes[i]), radius=radius, color=(255, 255, 255), thickness=3)


plt.imshow(img2)

plt.show()





