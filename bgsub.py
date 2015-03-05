import numpy as np
import cv2
import sys

#model history is how many frames does it look at for what the 'background' is
model_history = 60

#learning rate is how quickly objects that are still become background
learning_rate = 0.01

minimum_area = 50

background_ratio = 0.5
noise_strength = 1
n_mixtures = 5

#y_w,x_i,x_w
lookup_table_x = [[0,183,0],
[0,231,0.1],
[0,279,0.2],
[0,327,0.3],
[0,375,0.4],
[0,423,0.5],
[0,471,0.6],
[0,519,0.7],
[0,567,0.8],
[0,615,0.9],
[0,660,1],
[0.1,310,0],
[0.1,345,0.1],
[0.1,380,0.2],
[0.1,415,0.3],
[0.1,450,0.4],
[0.1,485,0.5],
[0.1,520,0.6],
[0.1,555,0.7],
[0.1,590,0.8],
[0.1,625,0.9],
[0.1,660,1],
[0.2,380,0],
[0.2,406,0.1],
[0.2,432,0.2],
[0.2,458,0.3],
[0.2,484,0.4],
[0.2,510,0.5],
[0.2,536,0.6],
[0.2,562,0.7],
[0.2,588,0.8],
[0.2,614,0.9],
[0.2,640,1],
[0.3,440,0],
[0.3,460,0.1],
[0.3,480,0.2],
[0.3,500,0.3],
[0.3,520,0.4],
[0.3,540,0.5],
[0.3,560,0.6],
[0.3,580,0.7],
[0.3,600,0.8],
[0.3,620,0.9],
[0.3,660,1],
[0.4,480,0],
[0.4,497.5,0.1],
[0.4,515,0.2],
[0.4,532.5,0.3],
[0.4,550,0.4],
[0.4,567.5,0.5],
[0.4,585,0.6],
[0.4,602.5,0.7],
[0.4,620,0.8],
[0.4,637.5,0.9],
[0.4,655,1],
[0.5,510,0],
[0.5,525,0.1],
[0.5,540,0.2],
[0.5,555,0.3],
[0.5,570,0.4],
[0.5,585,0.5],
[0.5,600,0.6],
[0.5,615,0.7],
[0.5,630,0.8],
[0.5,645,0.9],
[0.5,660,1],
[0.6,530,0],
[0.6,543,0.1],
[0.6,556,0.2],
[0.6,569,0.3],
[0.6,582,0.4],
[0.6,595,0.5],
[0.6,608,0.6],
[0.6,621,0.7],
[0.6,634,0.8],
[0.6,647,0.9],
[0.6,660,1],
[0.7,540,0],
[0.7,552,0.1],
[0.7,564,0.2],
[0.7,576,0.3],
[0.7,588,0.4],
[0.7,600,0.5],
[0.7,612,0.6],
[0.7,624,0.7],
[0.7,636,0.8],
[0.7,648,0.9],
[0.7,660,1],
[0.8,560,0],
[0.8,570,0.1],
[0.8,580,0.2],
[0.8,590,0.3],
[0.8,600,0.4],
[0.8,610,0.5],
[0.8,620,0.6],
[0.8,630,0.7],
[0.8,640,0.8],
[0.8,650,0.9],
[0.8,660,1],
[0.9,564,0],
[0.9,573.5,0.1],
[0.9,583,0.2],
[0.9,592.5,0.3],
[0.9,602,0.4],
[0.9,611.5,0.5],
[0.9,621,0.6],
[0.9,630.5,0.7],
[0.9,640,0.8],
[0.9,649.5,0.9],
[0.9,659,1],
[1,580,0],
[1,588,0.1],
[1,596,0.2],
[1,604,0.3],
[1,612,0.4],
[1,620,0.5],
[1,628,0.6],
[1,636,0.7],
[1,644,0.8],
[1,652,0.9],
[1,660,1]]

lookup_table_y = [[660,0],
[652,0.01],
[641.5,0.02],
[631,0.03],
[620.5,0.04],
[610,0.05],
[599.5,0.06],
[589,0.07],
[578.5,0.08],
[568,0.09],
[557.5,0.1],
[547,0.11],
[536.5,0.12],
[526,0.13],
[515.5,0.14],
[505,0.15],
[494.5,0.16],
[484,0.17],
[473.5,0.18],
[463,0.19],
[452.5,0.2],
[442,0.21],
[431.5,0.22],
[421,0.23],
[410.5,0.24],
[400,0.25],
[394,0.26],
[390.5,0.27],
[387,0.28],
[383.5,0.29],
[380,0.3],
[376.5,0.31],
[373,0.32],
[369.5,0.33],
[366,0.34],
[362.5,0.35],
[359,0.36],
[355.5,0.37],
[352,0.38],
[348.5,0.39],
[345,0.4],
[341.5,0.41],
[338,0.42],
[334.5,0.43],
[331,0.44],
[327.5,0.45],
[324,0.46],
[320.5,0.47],
[317,0.48],
[313.5,0.49],
[310,0.5],
[308,0.51],
[306,0.52],
[304,0.53],
[302,0.54],
[300,0.55],
[298,0.56],
[296,0.57],
[294,0.58],
[292,0.59],
[290,0.6],
[288,0.61],
[286,0.62],
[284,0.63],
[282,0.64],
[280,0.65],
[278,0.66],
[276,0.67],
[274,0.68],
[272,0.69],
[270,0.7],
[268,0.71],
[266,0.72],
[264,0.73],
[262,0.74],
[260,0.75],
[259.2,0.76],
[258.4,0.77],
[257.6,0.78],
[256.8,0.79],
[256,0.8],
[255.2,0.81],
[254.4,0.82],
[253.6,0.83],
[252.8,0.84],
[252,0.85],
[251.2,0.86],
[250.4,0.87],
[249.6,0.88],
[248.8,0.89],
[248,0.9],
[247.2,0.91],
[246.4,0.92],
[245.6,0.93],
[244.8,0.94],
[244,0.95],
[243.2,0.96],
[242.4,0.97],
[241.6,0.98],
[240.8,0.99],
[240,1]]



def calculateLookupY(y_i):
    y_w = None
    for i in range(0, len(lookup_table_y)-1):
        current_lookup = lookup_table_y[i]
        next_lookup = lookup_table_y[i+1]
        y_i_lookup = current_lookup[0]
        y_i_next_lookup = next_lookup[0]
        if  y_i_lookup > y_i > y_i_next_lookup:
            #print '********************FOND IT'
            y_w = current_lookup[1]
        else:
            #print '%s > %s > %s? NO' % (y_i_lookup, y_i, y_i_next_lookup)
            pass

    return y_w

def calculateLookupX(y_w, x_i):
    x_w = None
    if y_w == None:
        return None
    #select subset of lookup table
    lookup_table_subset = [ lookup for lookup in lookup_table_x if -0.05 < (lookup[0] - y_w) < 0.05 ]
    #print y_w
    #print lookup_table_subset
    for i in range(0, len(lookup_table_subset)-1):
        lookup = lookup_table_subset[i]
        next_lookup = lookup_table_subset[i+1]
        x_i_lookup = lookup[1]
        x_i_next_lookup = next_lookup[1]
        if x_i_lookup < x_i < x_i_next_lookup:
            x_w = lookup[2]
    return x_w

def histogramForRect(x, y, w, h, image):
    bins = np.arange(256).reshape(256,1)
    cropped_image = image[y:y+h, x:x+w]
    h = np.zeros((300,256,3))
    if len(cropped_image.shape) == 2:
        color = [(255,255,255)]
    elif cropped_image.shape[2] == 3:
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([cropped_image],[ch],None,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist)))
        cv2.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y

import string
import random
def id_generator(size=5, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))

def checkRatio(width, height):
    if height < 1.5 * width:
        return False
    else:
        return True

def processFrame(cap, kernel, fgbg, framecount):
    k = cv2.waitKey(30)
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame, learningRate=learning_rate)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #fgmask = cv2.dilate(fgmask, kernel, iterations=5)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)
    fgmask_copy = fgmask.copy()

    contours,hierarchy = cv2.findContours(fgmask_copy, 0, 2)
    
    x_top = 580
    y_top = 235

    x_bottom = 250
    y_bottom = 600

    #draw framecount
    cv2.putText(frame, 'frame: %s' % framecount, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    for contour in contours:

        #filter small boxes
        area = cv2.contourArea(contour)

        if area < minimum_area:
            continue

        '''
        #rotated rect
        rect = cv2.minAreaRect(contour)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame,[box],0,(0,0,255),2)
        '''

        #straight rect
        x,y,w,h = cv2.boundingRect(contour)

        if checkRatio(w, h) == False:
            continue

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        x_offset = 100
        y_offset = 100
        
        y_w = calculateLookupY(y+h)
        x_w = calculateLookupX(y_w, x)

        #draw histograms
        histogram = histogramForRect(x,y,w,h,frame)
        cv2.imshow('histogram', histogram)

        #draw foot position
        cv2.putText(frame, 'ic: %s, %s %s' % (x+w/2, y+h, id_generator()), (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
        cv2.putText(frame, 'wc: %s, %s' % (x_w, y_w), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
        
        #draw markers
        cv2.putText(frame, '- 0%', (660, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0)) 
        cv2.putText(frame, '- 100%', (660, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0)) 
        cv2.putText(frame, '- 50%', (660, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0)) 
        cv2.putText(frame, '- 75%', (660, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0)) 
        cv2.putText(frame, '- 25%', (660, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0)) 

    cv2.imshow('threshold',fgmask)
    cv2.imshow('frame',frame)
    '''
    #k = cv2.waitKey(30) & 0xff
    k = cv2.waitKey(30)


    #wait for space bar press
    if k == 32:
        continue
    else:
        k = cv2.waitKey(30)
    '''

cap = cv2.VideoCapture(str(sys.argv[1]))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

fgbg = cv2.BackgroundSubtractorMOG(model_history, n_mixtures, background_ratio, noise_strength)

framecount = 0

k = 32
while(1):
    if k == 32:
        processFrame(cap, kernel, fgbg, framecount)
        framecount = framecount + 1
    elif k == 27:
        break
    k = cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()





