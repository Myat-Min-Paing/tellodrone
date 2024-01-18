import pickle
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from djitellopy import tello
import time

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic
wCam, hCam = 640, 480
gesture_arr = ["Thumb_up","Thumb_down","Left","Right","Forward","Backward","Stop","Land","flip"]

drone_me = tello.Tello()
drone_me.connect()

def controllDrone(hand_sign):
    if hand_sign == gesture_arr[0]:
        cv2.putText(image, "Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        drone_me.send_rc_control(0, 0, 20, 0)
    elif hand_sign == gesture_arr[1]:
        cv2.putText(image, "Down", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        drone_me.send_rc_control(0, 0, -20, 0)
    elif hand_sign == gesture_arr[2]:
        cv2.putText(image, "Left", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        drone_me.send_rc_control(20, 0, 0, 0)
    elif hand_sign == gesture_arr[3]:
        cv2.putText(image, "Right", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        drone_me.send_rc_control(-20, 0, 0, 0)
    elif hand_sign == gesture_arr[4]:
            cv2.putText(image, "Fordward", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            drone_me.send_rc_control(0, 20, 0, 0)
    elif hand_sign == gesture_arr[5]:
            cv2.putText(image, "Backward", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            drone_me.send_rc_control(0, -20, 0, 0)
    elif hand_sign == gesture_arr[6]:
        cv2.putText(image, "Stop", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        drone_me.send_rc_control(0, 0, 0, 0)
    elif hand_sign == gesture_arr[7]:
            cv2.putText(image, "Land", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            drone_me.send_rc_control(0, 0, 0, 0)
            #drone_me.land()
    elif hand_sign == gesture_arr[8]:
            cv2.putText(image, "Flip", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            drone_me.send_rc_control(0, 0, 0, 0)
            #drone_me.flip_back()
    else:
        cv2.putText(image, "No Gesture", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        drone_me.send_rc_control(0, 0, 0, 0)

with open("hand_gesture.pkl", "rb") as f:
    model = pickle.load(f)

drone_me.streamon()
drone_me.takeoff()
drone_me.send_rc_control(0, 0, 25, 0)
time.sleep(1.2)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while drone_me.stream_on:
        frame = drone_me.get_frame_read().frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image,(640,480))
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        #2. Left hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                )

        # 3. Right Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        
        
        # Export coordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            
            #Extract Hand landmarks
            left_hand = results.right_hand_landmarks.landmark
            left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
            
            # Concate rows
            row = left_hand_row
            
            #Make Detections
            X = pd.DataFrame([row])
            hand_sign_class = model.predict(X)[0]
            hand_sign_prob = model.predict_proba(X)[0]
            
            #Grab ear coords
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [640,480]).astype(int))
            """cv2.rectangle(image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(body_language_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)"""
            cv2.putText(image, hand_sign_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(round(hand_sign_prob[np.argmax(hand_sign_prob)],2))
                        , (coords[0], coords[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            controllDrone(hand_sign_class)
        except:
            pass
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
#cap.release()
drone_me.streamoff()
cv2.destroyAllWindows()