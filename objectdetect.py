import cv2        

# img = cv2.imread('G:/Opencv/object detection/frnds.jpg') 

# capturing the image using webcam

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames=[]
classfile ='G:/Opencv/object detection/coco.names' # importing the name file
with open(classfile,'rt') as f :                       # taking the names from file and printing it
    classNames = f.read().rstrip('\n').split('\n')

# print(classNames)

configPath = 'G:/Opencv/object detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'    
weightsPath = 'G:/Opencv/object detection/frozen_inference_graph.pb'

net =cv2.dnn_DetectionModel(weightsPath,configPath) 
#  this all are the default values used for detection model
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)  

while True:
    success,img=cap.read()
    # sending the images to the model 
    classIds, confs, bbox = net.detect(img,confThreshold=0.5)  # confThreshold is 0.5 i.e the prediction is of 50% if the object gets less then 50%(0.5 thres) accurace then it will ignore it    
    print(bbox,classIds,confs)

    if len(classIds) !=0 :
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):  # created single forloop for all three at a single time
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)   # giving bounding box to img
            cv2.putText(img, classNames[classId-1] ,(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0),1)
            cv2.putText(img, str(round(confidence*100,2)) ,(box[0]+150,box[1]+100),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0),1)
        

    cv2.imshow("output",img)
    cv2.waitKey(1)



