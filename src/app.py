import cv2
import os

# Existing modules (DO NOT MODIFY)
from hand_detector import HandDetector, WebcamCapture
from gesture_recognizer import GestureRecognizer
from image_transformer import ImageTransformer


# =========================
# UI HELPERS
# =========================
def draw_rulebook(frame):
    """
    Draws a semi-transparent rulebook panel on the webcam feed.
    """
    overlay = frame.copy()

    # Panel rectangle
    x1, y1, x2, y2 = 10, 10, 300, 200
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 50, 50), -1)

    # Transparency
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    rules = [
        "GESTURE RULEBOOK",
        "----------------------",
        "Pinch        -> Zoom",
        "Tilt Hand    -> Rotate",
        "Move Hand    -> Pan",
        "Fist         -> Flip",
        "",
        "R -> Reset",
        "Q -> Quit"
    ]

    for i, text in enumerate(rules):
        cv2.putText(
            frame,
            text,
            (20, 35 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


def draw_center_marker(frame):
    """
    Optional visual reference for interaction center.
    """
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)

    cv2.circle(frame, center, 5, (0, 255, 255), -1)
    cv2.line(frame, (center[0] - 20, center[1]),
             (center[0] + 20, center[1]), (0, 255, 255), 1)
    cv2.line(frame, (center[0], center[1] - 20),
             (center[0], center[1] + 20), (0, 255, 255), 1)


# =========================
# IMAGE LOADING
# =========================
def load_image_from_user():
    """
    Load image from user input path.
    Falls back to default if invalid.
    """
    path = input("Enter image path (or press Enter for default): ").strip()

    if path and os.path.exists(path):
        print(f"Loaded image: {path}")
        return ImageTransformer(path)
    else:
        print("Using default generated image.")
        return ImageTransformer()


# =========================
# MAIN APPLICATION
# =========================
def main():
    print("🚀 Starting Gesture-Controlled Image App...")

    # Initialize modules
    detector = HandDetector()
    recognizer = GestureRecognizer()
    cap = WebcamCapture()

    transformer = load_image_from_user()

    print("Controls: Q = Quit | R = Reset")

    # Store last valid result (for "freeze when no hand")
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # === HAND DETECTION ===
        landmarks = detector.detect(frame)

        if landmarks:
            result = recognizer.recognize(
                landmarks,
                frame_width=cap.width,
                frame_height=cap.height
            )
            last_result = result
        else:
            result = last_result  # Freeze state if no hand

        # === APPLY TRANSFORMATIONS ===
        if result:
            transformed = transformer.apply_all(
                scale=result.scale_factor,
                rotation=result.rotation_angle,
                translation=result.translation,
                flip_horizontal=result.flip_horizontal
            )

            cv2.imshow("Transformed Image", transformed)

            # === DISPLAY INFO ===
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
                    (10, 250 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # === DRAW UI ===
        detector.draw_landmarks(frame, landmarks)
        draw_rulebook(frame)
        draw_center_marker(frame)

        cv2.imshow("Gesture Control (Webcam)", frame)

        # === CONTROLS ===
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            recognizer.reset()
            transformer.reset()
            last_result = None
            print("🔄 Reset everything")

        elif key == ord('i'):
            # Optional: load new image at runtime
            transformer = load_image_from_user()

    # Cleanup
    cap.release()
    detector.release()
    cv2.destroyAllWindows()
    print("✅ Exited cleanly.")


if __name__ == "__main__":
    main()