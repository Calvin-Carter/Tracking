import cv2
import os
import numpy as np

coords = []
dataset = []
data_labels = []
directory = 'D:\ProgrammeerOmgeving\Projects\Tracking\Data'

for filename in os.listdir(directory):
    dataset.append("../Data/"+filename)

with open('coordinates.txt') as f:
    for line in f:
        x,y = line.split()
        coords.append(np.array((int(x),int(y))))

def labelify(data, coords):
    for images, coord in zip(data,coords):
        img = cv2.imread(images)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                patch = img[y - 5: y + 6, x - 5: x + 6, :]
                patchCoords = np.array([x,y])
                dist = np.linalg.norm(coord - patchCoords)
                if dist < 20:
                    data_labels.append(patch)
                    patch[..., 2] = 255
        cv2.imshow("image", img)
        cv2.waitKey(0)

labelify(dataset, coords)



