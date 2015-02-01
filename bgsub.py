import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(str(sys.argv[1]))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
fgbg = cv2.BackgroundSubtractorMOG(1,5,0.5,1)

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=3)
    fgmask_copy = fgmask.copy()

    contours,hierarchy = cv2.findContours(fgmask_copy, 0, 2)
    
    for contour in contours:

        #filter small boxes
        area = cv2.contourArea(contour)

        if area < 1000:
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
        
        #draw foot position
        cv2.putText(frame, 'pos: %s, %s' % (x, y+h), (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0))

    cv2.imshow('threshold',fgmask)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

