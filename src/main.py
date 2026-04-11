import cv2
from hand_detector import HandDetector, WebcamCapture
from gesture_recognizer import GestureRecognizer

def main():
    detector = HandDetector()
    cap = WebcamCapture()
    recognizer = GestureRecognizer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Step 1: Detect hand
        landmarks = detector.detect(frame)

        # Step 2: Recognize gesture
        gesture = recognizer.recognize(landmarks)

        # Step 3: Draw landmarks
        detector.draw_landmarks(frame, landmarks)

        # Step 4: Display gesture
        if gesture:
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    detector.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()