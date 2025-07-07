import cv2
import mediapipe as mp
import time
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION  # or use FACEMESH_CONTOURS etc.

class faceMeshDetection():
    def __init__(self, max_num_faces=1, thickness=1, circle_radius=2):
        self.max_num_faces = max_num_faces
        self.thickness = thickness
        self.circle_radius = circle_radius

        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=self.max_num_faces)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=self.thickness, circle_radius=self.circle_radius)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        face_landmarks,
                        FACEMESH_TESSELATION,  # change to FACEMESH_CONTOURS for detailed outline
                        self.drawSpec,
                        self.drawSpec
                    )

                face = []
                for idx, landmark in enumerate(face_landmarks.landmark):
                    h, w, _ = img.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    face.append([idx, x, y])
                faces.append(face)

        return img, faces


def main():
    prev_time = 0
    video_capture = cv2.VideoCapture(0)
    faceMeshDetector = faceMeshDetection()

    while True:
        success, img = video_capture.read()
        if not success:
            break

        img, faces = faceMeshDetector.findFaceMesh(img)
        if faces:
            print(f"Faces detected: {len(faces)}")

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
