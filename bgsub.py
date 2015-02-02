import numpy as np
import cv2
import sys

def processFrame(cap, kernel, fgbg, framecount):
    k = cv2.waitKey(30)
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=5)
    fgmask_copy = fgmask.copy()

    contours,hierarchy = cv2.findContours(fgmask_copy, 0, 2)
    
    x_top = 580
    y_top = 235

    x_bottom = 250
    y_bottom = 600


    for contour in contours:

        #filter small boxes
        area = cv2.contourArea(contour)

        if area < 20:
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
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        x_offset = 100
        y_offset = 100
        
        x_w = x - y * 0.1
        y_w = 1.1 * y
        
        #draw foot position
        #cv2.putText(frame, 'ic: %s, %s' % (x, y+h), (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
        #cv2.putText(frame, 'wc: %s, %s' % (x_w, y_w), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
        
        #draw markers
        cv2.putText(frame, '- 0%', (660, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0)) 
        cv2.putText(frame, '- 100%', (660, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0)) 
        cv2.putText(frame, '- 50%', (660, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0)) 
        cv2.putText(frame, '- 75%', (660, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0)) 
        cv2.putText(frame, '- 25%', (660, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0)) 

        #draw framecount
        cv2.putText(frame, 'frame: %s' % framecount, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

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
fgbg = cv2.BackgroundSubtractorMOG(1,5,0.5,1)

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





