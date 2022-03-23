import cv2
import os

coords = []
dataset = []
directory = 'D:\ProgrammeerOmgeving\Projects\Tracking\Data'

for filename in os.listdir(directory):
    dataset.append("../Data/"+filename)

def mousePoints(event, x, y, flags, params):
    mouseclick = False
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseclick = True
        currentCoords = (x,y)
        coords.append(currentCoords)
        print(x,y)
        print(coords)
    if(mouseclick is True):
        cv2.destroyAllWindows()
        mouseclick = False

def showImages(data):
        for images in data:
            img = cv2.imread(images)
            cv2.imshow("images", img)
            cv2.setMouseCallback("images", mousePoints)
            cv2.waitKey(0)

cv2.waitKey(0)
showImages(dataset)

with open('coords.txt', 'w') as f:
    for item in coords:
        f.write(str(item) + "\n")
