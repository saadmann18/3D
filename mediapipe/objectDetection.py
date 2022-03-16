import cv2
import time
import numpy as np
import mediapipe as mp



mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils 

# Webcam
cap = cv2.VideoCapture(0)


with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.8,
                            model_name='Shoe') as objectron:


    while cap.isOpened():

        # Read in the image
        success, image = cap.read()

        # start time to calculate FPS
        start = time.time()

        #Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False

        # Process the image and find hands
        results = objectron.process(image)

        image.flags.writeable = True

        # Draw the hand annotations on the image.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)

        # End time
        end = time.time()
        totalTime = end-start

        # calculate the FPS for current frame detection
        fps = 1 / totalTime

        cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        
        cv2.imshow('MediaPipe Objectron', image)


        if cv2.waitKey(5) & 0xFF == 27:
            break

     

    cap.release()
    #cv2.destroyAllWindows()
