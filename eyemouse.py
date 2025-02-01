import cv2
import mediapipe as mp
import pyautogui

# Initialize camera
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get screen size
screen_w, screen_h = pyautogui.size()

while True:
    # Read frame from webcam
    success, frame = cam.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)  # Flip to mirror view
    frame_h, frame_w, _ = frame.shape

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Eye tracking - right eye (landmarks 474-478)
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            if id == 1:  # Landmark 475 - Main tracking point
                screen_x = int(screen_w * landmark.x)
                screen_y = int(screen_h * landmark.y)
                pyautogui.moveTo(screen_x, screen_y)

        # Blink detection - left eye (landmarks 145 & 159)
        left_eye_top = landmarks[145].y
        left_eye_bottom = landmarks[159].y

        # Draw circles on the left eye
        for landmark in [landmarks[145], landmarks[159]]:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        # If the difference is small, it's a blink -> Click event
        if (left_eye_bottom - left_eye_top) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)  # Prevent multiple clicks

    # Display frame
    cv2.imshow('Eye Controlled Mouse', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
