import numpy as np
import cv2 as cv
import uuid
import os


def round_nearest(x, a):
    return round(x / a) * a

if __name__ == '__main__':
    totalfiles = len(os.listdir('success'))
    for idx1,filename in enumerate(os.listdir('success')):
        # if(idx1 != 9000):
        #     continue
        print(idx1/totalfiles)
        file = 'success/'+filename
        label = file.split('.')[0].split('/')[1]


        im = cv.imread(file)
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        # cv.imshow('gray', imgray)
        # cv.waitKey(0)

        # edged = cv.Canny(thresh, 30, 200)
        # cv.imshow('canny', edged)
        # cv.waitKey(0)
        #blur = cv.GaussianBlur(imgray, (5, 5), 0)

        ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY_INV)
        #cv.imshow('thresh', thresh)
        #cv.waitKey(0)

        

        # contours, hierarchy = cv.findContours(
        #     thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        contours, hierarchy = cv.findContours(
            thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        #print("Number of Contours found = " + str(len(contours)))

        #newImg = cv.drawContours(im, contours, -1, (0, 255, 0), -1)
        #cv.imshow('Contours', newImg)
        #cv.waitKey(0)
        #cv.destroyAllWindows()


        total = 0

        images = []

        for idx, cont in enumerate(contours):
            x, y, width, height = cv.boundingRect(cont)
            # cv.imshow('Contours', thresh[y:y+height, x:x+width])
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            total += width


        for idx, cont in enumerate(contours):
            x, y, width, height = cv.boundingRect(cont)
            percentage = round_nearest(width/total,0.2)
            #percentage = round(width/total, 1)
            #print(percentage)
            parts = int(percentage*5)
            #print(parts)
            if parts > 1:
                part_w = int(width/parts)
                for part in range(parts):

                    roi = thresh[y:y+height, x+part_w*part:x+part_w*(part+1)]
                    images.append((x+part_w*part,roi))
                    #cv.imwrite(str(count)+".png", roi)
            else:
                roi = thresh[y:y+height, x:x+width]
                images.append((x, roi))
                #cv.imwrite(str(count)+".png", roi)

        images.sort(key=lambda x: x[0])
        #print('Hay',len(images),'letras')
        cont_dict = { k:0 for k in label }
        for idx,image in enumerate(images):
            try:
                resized = cv.resize(image[1], (15, 30))
                folder = "parts/"+label[idx]
                if not os.path.exists(folder):
                    os.mkdir(folder)
                cont_dict[label[idx]]+=1
                route = "parts/"+label[idx]+"/"+label + \
                    '-'+str(cont_dict[label[idx]])+".png"
                #print(route)
                cv.imwrite(route, resized)
            except Exception as identifier:
                print(identifier)
                #print(label)
                #print(idx)
                #cv.waitKey(0)


        #break

        
