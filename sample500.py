import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from alive_progress import alive_bar
import time
import random
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, 0.6) * 255.0, 0, 255)

def thre(image1, image2):
    global lookUpTable
    

    daytime_img = cv.imread(image1)
    night_img = cv.imread(image2)
# Denoise
    night_img = cv.fastNlMeansDenoising(night_img)
    daytime_img = cv.fastNlMeansDenoising(daytime_img)

# Gamma correction
    night_img = cv.LUT(night_img, lookUpTable)
    daytime_img = cv.LUT(daytime_img, lookUpTable)

# Harris corner
    night_gray = cv.cvtColor(night_img, cv.COLOR_BGR2GRAY)
    daytime_gray = cv.cvtColor(daytime_img, cv.COLOR_BGR2GRAY)

# Find corners
    kp_night = cv.goodFeaturesToTrack(night_gray, 5000, 0.01, 10)
    kp_daytime = cv.goodFeaturesToTrack(daytime_gray, 5000, 0.01, 10)
    l_night = []
    for item in kp_night:
        l_night.append(item[0])
    kp_night = np.array(l_night)
    kp_night = cv.KeyPoint_convert(kp_night)
    l_daytime = []
    for item in kp_daytime:
        l_daytime.append(item[0])
    kp_daytime = np.array(l_daytime)
    kp_daytime = cv.KeyPoint_convert(kp_daytime)

# SIFT
    sift = cv.SIFT_create()
    kp_night, des_night = sift.compute(night_gray, kp_night)
    kp_daytime, des_daytime = sift.compute(daytime_gray, kp_daytime)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_night,des_daytime, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    return(len(good))
training = os.listdir("./train")
training = [i for i in training if "DS" not in i]
def getimages(path):
    path = "./train/" + path
    images = os.listdir(path)
    images = [i for i in images if "DS" not in images]
    pairs = [(a, b) for idx, a in enumerate(images) for b in images[idx + 1:]]
    if len(pairs) >= 500:
        pairs = random.sample(pairs, 500)
    return pairs
overall_threshold = []
for i in training:
    threshold = []
    pairs = getimages(i)
    path = "./train/"+i+"/"
    with alive_bar(len(pairs)) as bar:
        for j in pairs:
            threshold.append(thre(path+j[0], path+j[1]))
            time.sleep(0.001)
            bar()
    overall_threshold.append(threshold)
## Becasue all taining images are from the same images, so we can set min threshold as overall_threshold.
th = min([i for j in overall_threshold for i in j])
print(th)

## check the accuracy for testing data.
def gettest(path):
    path = "./test/" + path
    images = os.listdir(path)
    images = [i for i in images if "DS" not in images]
    pairs = [(a, b) for idx, a in enumerate(images) for b in images[idx + 1:]]
    if len(pairs) >= 500:
        pairs = random.sample(pairs, 500)
    return pairs
testing = os.listdir("./test")
testing = [i for i in testing if "DS" not in i]
yes = 0
no = 0
for i in testing:
    path = "./test/"+i+"/"
    test_pairs = gettest(i)
    with alive_bar(len(test_pairs)) as bar:
        for j in test_pairs:
            if thre(path+j[0],path+j[1])>=th:
                yes +=1
            else:
                no +=1
            time.sleep(0.001)
            bar()
accuracy = yes/(yes+no)
print(accuracy)
overall_threshold2 = [sum(i)/len(i) for i in overall_threshold]
textfile = open("result500.txt", "w")
for i in overall_threshold2:
    textfile.write(str(i) + "\n")
textfile.close()
