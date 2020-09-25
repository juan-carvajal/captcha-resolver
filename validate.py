from tensorflow import keras
from keras.preprocessing import image as imageloader
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import random
from process import round_nearest
model = keras.models.load_model('model.h5')

totalfiles = len(os.listdir('success'))
good_guess=0
num_tests=500
totalfiles=num_tests
#for idx1, filename in enumerate(os.listdir('success')):
for idx1 in range(num_tests):
    filename = random.choice(os.listdir('success'))
    print("{:.2%}".format(idx1/totalfiles))
    file = 'success/'+filename
    label = file.split('.')[0].split('/')[1]

    im = cv.imread(file)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY_INV)

    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    total = 0

    images = []

    for idx, cont in enumerate(contours):
        x, y, width, height = cv.boundingRect(cont)
        total += width

    for idx, cont in enumerate(contours):
        x, y, width, height = cv.boundingRect(cont)
        percentage = round_nearest(width/total, 0.2)
        parts = int(percentage*5)
        if parts > 1:
            part_w = int(width/parts)
            for part in range(parts):

                roi = thresh[y:y+height, x+part_w*part:x+part_w*(part+1)]
                images.append((x+part_w*part, roi))
        else:
            roi = thresh[y:y+height, x:x+width]
            images.append((x, roi))

    images.sort(key=lambda x: x[0])
    for idx, image in enumerate(images):
        try:
            resized = cv.resize(image[1], (15, 30))
            cv.imwrite('temp/'+str(idx)+'.png', resized)
        except Exception as ex:
            print(ex)


    guess = ''
    cats = list('123456789ABCDEFHKMNPRSTUVWXYZ')
    for idx, image in enumerate(images):
        path = 'temp/'+str(idx)+'.png'
        test_image = imageloader.load_img(path, target_size=(30, 15))
        test_image = imageloader.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)[0]
        #index = np.where(result == 1)
        index=int(np.argmax(result))
        #print('index',index)
        #print('result', result)
        #print()
        guess += cats[index]
    if(guess == filename.split('.')[0].upper()):
        good_guess+=1
print('Guess acc:', "{:.2%}".format(good_guess/totalfiles))
