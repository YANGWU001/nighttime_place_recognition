import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from alive_progress import alive_bar
import time
import random
import itertools
import pandas as pd
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
#overall_threshold = []
#for i in training:
    #threshold = []
    #pairs = getimages(i)
    #path = "./train/"+i+"/"
    #with alive_bar(len(pairs)) as bar:
        #for j in pairs:
            #threshold.append(thre(path+j[0], path+j[1]))
            #time.sleep(0.001)
            #bar()
    #overall_threshold.append(threshold)
## Becasue all taining images are from the same images, so we can set min threshold as overall_threshold.
#th = min([i for j in overall_threshold for i in j])
#print(th)
## check the accuracy for testing data.
def gettest(path):
    path = "./test/" + path
    images = os.listdir(path)
    images = [path+"/"+i for i in images if "DS" not in images]
    pairs = [(a, b) for idx, a in enumerate(images) for b in images[idx + 1:]]
    random.seed(10)
    pairs = random.sample(pairs, 250)
    return pairs
testing = os.listdir("./test")
testing = [i for i in testing if "DS" not in i]
g1 = []
for i in testing:
    aa = gettest(i)
    g1 += aa
def unpair(path):
    a = os.listdir(path)
    a = [i for i in a if "DS" not in i]
    b1 = os.listdir(path+"/" +a[0])
    b1 = [path+"/"+a[0]+"/"+i for i in b1 if "DS" not in i]
    b2 = os.listdir(path + "/" + a[1])
    b2 = [path+"/"+a[1]+"/"+i for i in b2 if "DS" not in i]
    random.seed(40)
    unpaired = random.sample(set(itertools.product(b1,b2)),500)
    return unpaired
g0 = unpair("./test")
ths = [57,60,59,112,113,84,85,86,103,101,105]
result = pd.DataFrame(index = range(len(ths)), columns = ["threshold", "accuracy","recall", "precision", "F1"])

result["threshold"] = ths
g1_result =[]
with alive_bar(500) as bar:
    for i in g1:
        g1_result.append(thre(i[0],i[1]))
        time.sleep(0.001)
        bar()
g0_result = []
with alive_bar(500) as bar:
    for i in g0:
        g0_result.append(thre(i[0],i[1]))
        time.sleep(0.001)
        bar() 
for t in range(len(ths)):
    tp = len([i for i in g1_result if i >=ths[t]])
    fp = len([i for i in g0_result if i >=ths[t]])
    fn = len([i for i in g1_result if i <ths[t]])
    tn = len([i for i in g0_result if i <ths[t]])
    accuracy = (tp+tn)/1000
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    result.iloc[t,1:] = [accuracy,recall,precision,f1]
result.to_csv("threshold_result.csv", index = False)
