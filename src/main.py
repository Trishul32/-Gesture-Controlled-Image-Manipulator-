import cv2

# Import your modules
from hand_detector import HandDetector, WebcamCapture
from gesture_recognizer import GestureRecognizer


def main():
    print("Starting Gesture Control System...")

    # Initialize components
    detector = HandDetector()
    recognizer = GestureRecognizer()
    cap = WebcamCapture()

    print("Press 'q' to quit, 'r' to reset transformations")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame for natural interaction
        frame = cv2.flip(frame, 1)

        # Step 1: Detect hand
        landmarks = detector.detect(frame)

        # Step 2: Recognize gestures
        if landmarks:
            result = recognizer.recognize(
                landmarks,
                frame_width=cap.width,
                frame_height=cap.height
            )

            # Step 3: Display results
            info = [
                f"Gesture: {result.gesture_name}",
                f"Scale: {result.scale_factor:.2f}x",
                f"Rotation: {result.rotation_angle:.1f} deg",
                f"Translation: {result.translation}",
                f"Flip: {result.flip_horizontal}"
            ]

            for i, text in enumerate(info):
                cv2.putText(
                    frame,
                    text,
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
        else:
            cv2.putText(
                frame,
                "No Hand Detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Step 4: Draw landmarks
        detector.draw_landmarks(frame, landmarks)

        # Step 5: Show frame
        cv2.imshow("Gesture Control", frame)

        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recognizer.reset()
            print("🔄 Reset transformations")

    # Cleanup
    cap.release()
    detector.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
