import mediapipe as mp
import cv2
import math
import numpy as np
from djitellopy import tello
import time

drone_me = tello.Tello()
drone_me.connect()

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
pid = [0.4, 0.4, 0]
p_error = 0

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

def controllDrone(area,nose_center,w,pid,p_error):
    global forward_backward
    error = (nose_center * w) - w // 2
    if area <= 3000 and area != 0:
        forward_backward = 20
    if 3000 < area < 5000:
        forward_backward = 0
    if area >= 5000:
        forward_backward = -20

    if  nose_center < 0.3:
        speed = 20
    elif round(nose_center) > 0:
        speed = -20
    else:
        speed = 0
    drone_me.send_rc_control(0, forward_backward, 0, speed)
    return error

#cap = cv2.VideoCapture(1)
drone_me.streamon()
drone_me.takeoff()
drone_me.send_rc_control(0, 0, 25, 0)
time.sleep(1.2)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while drone_me.streamon:
        #ret, frame = cap.read()
        frame = drone_me.get_frame_read().frame
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Left hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Right Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        if results.face_landmarks:
            
            face = results.face_landmarks.landmark
            
            landmarks_x = list(landmark.x * image.shape[1] for landmark in face)
            landmarks_y = list(landmark.y * image.shape[0] for landmark in face)
            coordinates = np.column_stack((landmarks_x, landmarks_y))

            # Apply the Shoelace formula to calculate the area
            area = 0.5 * np.abs(np.dot(coordinates[:, 0], np.roll(coordinates[:, 1], 1)) - np.dot(coordinates[:, 1], np.roll(coordinates[:, 0], 1)))
            face_area = int(area)
            
            nose_landmark = results.face_landmarks.landmark[4]
            center_x = nose_landmark.x * image.shape[1]
            center_y = nose_landmark.y * image.shape[0]
            nose_z = nose_landmark.z
            cv2.circle(image, (int(center_x), int(center_y)), 5, (0, 0, 255), cv2.FILLED)
            cv2.putText(image,f"face area:{face_area}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image,f"Nose centerx:{nose_landmark.x} ", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            p_error = controllDrone(face_area,nose_landmark.x,image.shape[1],pid,p_error)
        
        w = image.shape[1]  
        cv2.putText(image,f"Battery:{drone_me.get_battery()}%", (image.shape[0],50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
#cap.release()
drone_me.streamoff()
cv2.destroyAllWindows()