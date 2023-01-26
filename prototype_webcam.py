import cv2
import serial
import math
import time
import numpy as np

#Load OpenCV Deep Neural Network
net = cv2.dnn.readNet("dnn_model/yolov4.weights", "dnn_model/yolov4.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale = 1/255)

#Load Class Lists into Classes Array
classes = []
with open ('dnn_model/classes.txt', 'r') as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

#Initialize Video Capture & Video Size
cap = cv2.VideoCapture(0)
cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)

# Initialize Frame Count
count = 0
center_points_prev_frame = []

#Initialize List of Tracked Objects and Ids
tracking_objects = {}
#Start counting track id at 0
track_id = 0

#List Visit of Region 1, 2, 3
region_A_ids = set()
region_B_ids = set()

#Initialize Arduino Serial
arduino = serial.Serial('COM3', 9600, timeout=1)

pTime = 0

while True:
    #Get frames
    ret, frame = cap.read()
    count += 1
    if not ret:
        break
    
    height, width, _ = frame.shape

    cTime = time.time()

    center_points_cur_frame = []
    #Initial and Display Region of Interest
    region_A = [(0,0),(0,480),(213,370),(213,110)]
    for area in [region_A]:
        cv2.polylines(frame, [np.array(area, np.int32)],True , (15, 220, 10), 3)
    
    region_B =  [(426,110),(640,0), (640,480), (426,370)]
    for area in [region_B]:
        cv2.polylines(frame, [np.array(area, np.int32)], True, (15, 220, 10), 3)
    
    #Initialize Object Detection
    (class_ids, scores, boxes) = model.detect(frame, nmsThreshold=0.6, confThreshold=0.6)
    for (class_id, score, box) in zip(class_ids, scores, boxes):
        
        #Assign Box Coordinate from Boxes from Model
        (x, y, w, h) = box
        class_name = classes[class_id]

        #Set Only Person Class will be Detected
        if class_name == 'person':
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            #center_points.append((cx, cy))
            center_points_cur_frame.append((cx, cy))
            #Show ClassID in Detected Object
            cv2.rectangle(frame, (x, y-30), (x+85, y),(42, 219, 151), -1)
            cv2.putText(frame,class_name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 255, 255), 2)
    
    if count <= 2:
        #Calculate distance of center point of bounding box from detected object,   from current & previous frame
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                #If distance is less than 20 pixels
                if distance < 70:
                    #tracking objects {0: (cx, cy)}
                    tracking_objects[track_id] = pt
                    track_id += 1 
                    
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 70:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1
    
    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (42, 219, 151), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (255, 255, 255), 2)
        
        #Initialize Store Ids and Test Center Dot Entering Region
        inside_region_A = cv2.pointPolygonTest(np.array(region_A),pt, False)
        inside_region_B = cv2.pointPolygonTest(np.array(region_B),pt, False)

        if inside_region_A == 1:
            region_A_ids.add(object_id)
            total_person = str(len(region_A_ids)+len(region_B_ids))
            arduino.write(total_person.encode())
            time.sleep(0.5)
            
        if inside_region_B == 1:
            region_B_ids.add(object_id)
            total_person = str(len(region_A_ids)+len(region_B_ids))
            arduino.write(total_person.encode())
            time.sleep(0.5)
    
    fps = 1/ (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
            
    print("=========================")
    print("-------------------------")
    print("ID Tracked in Region 1:", str(region_A_ids))
    print("Total Person in Region 1:", len(region_A_ids))
    print(str(region_A_ids))

    print("-------------------------")
    print("ID Tracked in Region 2:", str(region_B_ids))
    print("Total Person in Region 2:", len(region_B_ids))
    print(str(region_B_ids))

    print("\nTotal All Customers in All Regions: ",len(region_A_ids)+len(region_B_ids))

    print("-------------------------")
    
    print("CUR FRAME LEFT PTS")
    print(center_points_cur_frame)
    print("=========================")

    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break
 
#Show Video Footage
cap.release()
cv2.destroyWindow() 