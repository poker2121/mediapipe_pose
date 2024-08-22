import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose  

cap = cv2.VideoCapture(0)

with mp_pose.Pose (min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:

 while cap.isOpened():
  ret, frame = cap.read()

  image = cv2.cvtColor (frame, cv2.COLOR_BGR2RGB)
  

  results = pose.process(image) 
  image = cv2.cvtColor (image, cv2.COLOR_RGB2BGR)

  mp_drawing.draw_landmarks(
    image,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=4, circle_radius=2),#point in joint
    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=4)#lines
 
)

  new_width = 1000
  new_height = 800
  image = cv2.resize(image, (new_width, new_height))

  cv2.imshow('pose estimation', image)
  if cv2.waitKey(1) == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
