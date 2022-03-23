import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

trained_coords = []
train_data = []
directory = 'D:\ProgrammeerOmgeving\Projects\Tracking\Data'
testimg = 'D:\ProgrammeerOmgeving\Projects\Tracking/penis.jpg'

for filename in os.listdir(directory):
    train_data.append("../Data/"+filename)

with open('coordinates.txt') as f:
    for line in f:
        x,y = line.split()
        trained_coords.append(np.array((int(x),int(y))))

def labelify(data, coords): #loopt door de foto heen
    patches = []
    labels = []
    for images, coord in zip(data,trained_coords):
        img = cv2.imread(images) #Basic Image
        for y in range(0, img.shape[0], 10):
            for x in range(0, img.shape[1], 10):
                patch = img[y: y + 10, x: x + 10] #size of patch/ROI
                patches.append(patch)
                patch_coords = np.array([(x + 5),(y + 5)])
                dist_to_coord = np.linalg.norm(coord - patch_coords)
                if dist_to_coord < 30:
                    labels.append("Nose")
                    patch[..., 2] = 255
                else:
                    labels.append("Other")
        #cv2.imshow("image", img)
        #cv2.waitKey(0)
    return np.array(patches), np.array(labels)

def test(svm, data):
    img = cv2.imread(data) #Basic Image0
    for y in range(0, img.shape[0], 10):
        for x in range(0, img.shape[1], 10):
            patch = img[y: y + 10, x: x + 10]
            nx, ny, rgb = patch.shape
            patch_2d = patch.reshape((nx*ny*rgb))
            patch_2d = patch_2d.reshape(1,-1)
            result = svm.predict(patch_2d)
            if result == "Nose":
                print("NOSE!!!!")
                patch[..., 2] = 255
    cv2.imshow("image", img)
    cv2.waitKey(0)

patches, labels = labelify(train_data, trained_coords)
nsamples, nx, ny, rgb = patches.shape
patches_2d = patches.reshape((nsamples,nx*ny*rgb))
X_train, X_test, y_train, y_test = train_test_split(patches_2d, labels, test_size=0.70, random_state=1)
svm = svm.SVC(kernel="rbf")#kernal used in paper
svm.fit(X_train, y_train)
test(svm, testimg)
#print(svm.score(X_test, y_test) * 100)

