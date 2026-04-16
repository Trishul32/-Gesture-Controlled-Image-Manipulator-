import cv2
import sys

from hand_detector import HandDetector, WebcamCapture
from gesture_recognizer import GestureRecognizer
from image_transformer import ImageTransformer


def main(image_path=None):
    print("🚀 Gesture-Controlled Image Manipulator")

    # Initialize modules
    detector = HandDetector()
    recognizer = GestureRecognizer()
    cap = WebcamCapture()

    # Load user image OR fallback
    transformer = ImageTransformer(image_path)

    print("Controls:")
    print(" - Pinch → Zoom")
    print(" - Tilt → Rotate")
    print(" - Move hand → Pan")
    print(" - Fist → Flip")
    print(" - Press 'r' → Reset")
    print(" - Press 'q' → Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Detect hand
        landmarks = detector.detect(frame)

        if landmarks:
            # Recognize gesture
            result = recognizer.recognize(
                landmarks,
                cap.width,
                cap.height
            )

            # Apply transformation to image
            transformed_img = transformer.apply_all(
                scale=result.scale_factor,
                rotation=result.rotation_angle,
                translation=result.translation,
                flip_horizontal=result.flip_horizontal
            )

            # Display gesture info
            info = [
                f"Gesture: {result.gesture_name}",
                f"Scale: {result.scale_factor:.2f}",
                f"Rotation: {result.rotation_angle:.1f}",
                f"Translation: {result.translation}",
                f"Flip: {result.flip_horizontal}"
            ]

            for i, text in enumerate(info):
                cv2.putText(frame, text, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

        else:
            transformed_img = transformer.apply_all()
            cv2.putText(frame, "No Hand Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        # Draw landmarks
        detector.draw_landmarks(frame, landmarks)

        # Show both windows
        cv2.imshow("Webcam Feed", frame)
        cv2.imshow("Transformed Image", transformed_img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            recognizer.reset()
            transformer.reset()
            print("🔄 Reset!")

    cap.release()
    detector.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Allow user to pass image path
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(image_path)