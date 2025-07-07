import cv2
import mediapipe as mp
import time

prev_time = 0
vc = cv2.VideoCapture(0)

mpfacemesh = mp.solutions.face_mesh
mpdraw = mp.solutions.drawing_utils
facemesh = mpfacemesh.FaceMesh(max_num_faces=2)
drawspec = mpdraw.DrawingSpec(thickness = 1, circle_radius=2)

while True:
    _, img = vc.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = facemesh.process(imgRGB)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks: # number of faces
            # mpDraw.draw_landmarks(img, face_landmarks)
            mpdraw.draw_landmarks(
                img, 
                face_landmarks, 
                mpfacemesh.FACE_CONNECTIONS,
                drawspec, drawspec) # joins the points outlining the face, eyebrows, eyes, nose and lips

            for idx, landmark in enumerate(face_landmarks.landmark):  #468 points
                height, width, _ = img.shape
                x,y = int(landmark.x * width), int(landmark.y * height)
                print(idx, x,y)


    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time

    cv2.putText(img, f'FPS:{int(fps)}', (10,70), cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0),3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
    