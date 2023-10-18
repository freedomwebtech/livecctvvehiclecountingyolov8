import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
from tracker import*
model=YOLO('yolov8s.pt')

stream = CamGear(source='https://www.youtube.com/watch?v=En_3pkxIJRM', stream_mode = True, logging=True).start() # YouTube Video URL as input

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)




my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0
tracker =Tracker()


while True:    
    frame = stream.read()   
    count += 1
    if count % 2 != 0:
        continue


    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id1=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
       
        cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
        cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2) 
        cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                 
   


    cv2.imshow("RGB", frame)
    

    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()


