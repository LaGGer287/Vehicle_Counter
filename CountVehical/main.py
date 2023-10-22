import cv2
import torch
from tracker import *
import numpy as np
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('video.mp4')


def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

count = 0

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)
area1 = [(226,329),(173,367),(431,367),(448,329)]
area_1 = set()
area2 = [(561,329),(567,367),(829,367),(780,329)]
area_2 = set()

tracker = Tracker()
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(1020,500))
    results = model(frame)
    list = []
    for index,rows in results.pandas().xyxy[0].iterrows():
        x = int(rows['xmin'])
        y = int(rows['ymin'])
        x1 = int(rows['xmax'])
        y1 = int(rows['ymax'])
        b = str(rows['name'])
        list.append([x,y,x1,y1])
    idx_box = tracker.update(list)
    for bbox in idx_box:
        x2,y2,x3,y3,id = bbox
        cv2.putText(frame,str(id),(x2,y2),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),1)
        cv2.rectangle(frame,(x2,y2),(x3,y3),(0,0,255),2)
        cv2.circle(frame,(x3,y3),4,(0,255,0),-1)
        results = cv2.pointPolygonTest(np.array(area1,np.int32),((x3,y3)),False)
        results1 = cv2.pointPolygonTest(np.array(area2,np.int32),((x3,y3)),False)
        if results > 0:
            area_1.add(id)
        if results1 > 0:
            area_2.add(id)


    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,255),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,255),2)
    a1 = len(area_1)
    cv2.putText(frame,"So luong xe di xuong: " + str(a1),(391,52),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    a2 = len(area_2)
    cv2.putText(frame,"So luong xe di len: " + str(a2),(391,90),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.imshow('FRAME',frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
    
    
