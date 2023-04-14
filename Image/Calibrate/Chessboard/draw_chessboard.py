import cv2 
import numpy as np

# 9*7
# width = 450
# height = 350
# length = 50

# 6*6
width = 300
height = 300
length = 50

image = np.zeros((width,height),dtype = np.uint8)
print(image.shape[0],image.shape[1])

for j in range(height):
    for i in range(width):
        if((int)(i/length) + (int)(j/length))%2:
            image[i,j] = 255;

cv2.imwrite("./chessboard.jpg",image)