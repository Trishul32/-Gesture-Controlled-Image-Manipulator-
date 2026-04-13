<<<<<<< HEAD
"""
Gesture-Controlled Image Manipulator - Main Application
Real-Time Hand Gesture Based Image Transformations using OpenCV and MediaPipe

This is the main entry point that integrates all modules:
    - HandDetector (Person A): Webcam capture and hand landmark detection
    - GestureRecognizer (Person B): Gesture interpretation
    - ImageTransformer (Person C): Image transformations

Controls:
    q - Quit
    r - Reset all transformations
    c - Recalibrate gestures
    s - Save screenshot
    f - Toggle fullscreen

Gestures:
    Pinch (thumb + index): Zoom in/out
    Tilt hand: Rotate image
    Move hand: Pan image
    Make fist: Flip image horizontally
"""

import cv2
import numpy as np
import time
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hand_detector import HandDetector, WebcamCapture
from gesture_recognizer import GestureRecognizer, GestureResult
from image_transformer import ImageTransformer


class GestureControlApp:
    """Main application class for gesture-controlled image manipulation."""
    
    def __init__(
        self,
        image_path: str = None,
        camera_id: int = 0,
        window_width: int = 1280,
        window_height: int = 720
    ):
        """
        Initialize the application.
        
        Args:
            image_path: Path to image to manipulate (None = use test image)
            camera_id: Webcam device ID
            window_width: Combined display window width
            window_height: Combined display window height
        """
        self.window_width = window_width
        self.window_height = window_height
        self.running = False
        
        # Initialize modules
        print("Initializing Gesture-Controlled Image Manipulator...")
        print()
        
        try:
            print("[1/4] Initializing Hand Detector...")
            self.detector = HandDetector(
                max_hands=1,
                detection_confidence=0.7,
                tracking_confidence=0.7
            )
            print("      OK")
        except Exception as e:
            print(f"      FAILED: {e}")
            raise
        
        try:
            print("[2/4] Initializing Webcam...")
            self.webcam = WebcamCapture(camera_id=camera_id)
            print(f"      OK ({self.webcam.width}x{self.webcam.height})")
        except Exception as e:
            print(f"      FAILED: {e}")
            self.detector.release()
            raise
        
        print("[3/4] Initializing Gesture Recognizer...")
        self.recognizer = GestureRecognizer(
            smoothing_factor=0.3,
            scale_sensitivity=1.0,
            rotation_sensitivity=1.0,
            translation_sensitivity=1.0
        )
        print("      OK")
        
        print("[4/4] Initializing Image Transformer...")
        self.transformer = ImageTransformer(
            image_source=image_path,
            output_size=(400, 400)
        )
        print("      OK")
        
        # Available images for switching
        self.available_images = []
        image_files = [
            ("assets/3d_scene.png", "3D Scene"),
            ("assets/3d_cube.png", "3D Cube"),
            ("assets/3d_sphere.png", "3D Sphere"),
            ("assets/3d_pyramid.png", "3D Pyramid"),
        ]
        for path, name in image_files:
            if os.path.exists(path):
                self.available_images.append((path, name))
        self.current_image_index = 0
        
        print()
        print("=" * 60)
        print("  READY! Show your hand to control the image.")
        print("=" * 60)
        print()
        self._print_controls()
        
        # State
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.last_gesture = GestureResult()
        self.paused = False
    
    def _print_controls(self):
        """Print control instructions."""
        print("CONTROLS:")
        print("  q - Quit")
        print("  r - Reset transformations")
        print("  c - Recalibrate")
        print("  s - Save screenshot")
        print("  1-4 - Switch 3D image")
        print()
        print("GESTURES:")
        print("  Pinch fingers  → Zoom in/out")
        print("  Tilt hand      → Rotate image")
        print("  Move hand      → Pan image")
        print("  Make fist      → Flip image")
        print()
        
        if self.available_images:
            print("AVAILABLE IMAGES:")
            for i, (path, name) in enumerate(self.available_images, 1):
                print(f"  {i} - {name}")
            print()
    
    def run(self):
        """Main application loop."""
        self.running = True
        
        while self.running:
            # Capture frame
            ret, frame = self.webcam.read()
            if not ret:
                print("Failed to read from webcam")
                break
            
            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Detect hand
            landmarks = self.detector.detect(frame)
            
            # Recognize gesture
            if landmarks:
                gesture = self.recognizer.recognize(landmarks, w, h)
                self.last_gesture = gesture
            else:
                gesture = GestureResult(
                    hand_detected=False,
                    flip_horizontal=self.last_gesture.flip_horizontal,
                    scale_factor=self.last_gesture.scale_factor,
                    rotation_angle=self.last_gesture.rotation_angle,
                    translation=self.last_gesture.translation
                )
            
            # Apply transformations
            transformed = self.transformer.apply_all(
                scale=gesture.scale_factor,
                rotation=gesture.rotation_angle,
                translation=gesture.translation,
                flip_horizontal=gesture.flip_horizontal
            )
            
            # Draw landmarks on webcam view
            self.detector.draw_landmarks(frame, landmarks)
            
            # Calculate FPS
            self._update_fps()
            
            # Create combined display
            display = self._create_display(frame, transformed, gesture)
            
            # Show
            cv2.imshow("Gesture-Controlled Image Manipulator", display)
            
            # Handle input
            self._handle_input()
        
        self._cleanup()
    
    def _update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()
    
    def _create_display(
        self, 
        webcam_frame: np.ndarray, 
        transformed: np.ndarray,
        gesture: GestureResult
    ) -> np.ndarray:
        """
        Create combined display with webcam and transformed image side by side.
        """
        # Target dimensions
        panel_height = 480
        webcam_width = 640
        image_width = 400
        info_width = 240
        
        total_width = webcam_width + image_width + info_width
        
        # Create canvas
        canvas = np.zeros((panel_height, total_width, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)  # Dark gray background
        
        # Resize webcam frame
        webcam_resized = cv2.resize(webcam_frame, (webcam_width, panel_height))
        canvas[0:panel_height, 0:webcam_width] = webcam_resized
        
        # Resize transformed image
        img_h, img_w = transformed.shape[:2]
        scale = min(image_width / img_w, panel_height / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        img_resized = cv2.resize(transformed, (new_w, new_h))
        
        # Center transformed image
        x_offset = webcam_width + (image_width - new_w) // 2
        y_offset = (panel_height - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
        
        # Draw info panel
        info_x = webcam_width + image_width
        self._draw_info_panel(canvas, info_x, info_width, panel_height, gesture)
        
        # Draw separator lines
        cv2.line(canvas, (webcam_width, 0), (webcam_width, panel_height), (60, 60, 60), 2)
        cv2.line(canvas, (webcam_width + image_width, 0), 
                (webcam_width + image_width, panel_height), (60, 60, 60), 2)
        
        # Draw labels
        cv2.putText(canvas, "WEBCAM", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(canvas, "TRANSFORMED", (webcam_width + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        return canvas
    
    def _draw_info_panel(
        self, 
        canvas: np.ndarray, 
        x: int, 
        width: int, 
        height: int,
        gesture: GestureResult
    ):
        """Draw information panel on the right side."""
        y_pos = 30
        line_height = 28
        
        # Title
        cv2.putText(canvas, "INFO", (x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y_pos += line_height + 10
        
        # FPS
        fps_color = (0, 255, 0) if self.fps >= 25 else (0, 255, 255) if self.fps >= 15 else (0, 0, 255)
        cv2.putText(canvas, f"FPS: {self.fps:.1f}", (x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        y_pos += line_height
        
        # Hand status
        if gesture.hand_detected:
            status_color = (0, 255, 0)
            status_text = "HAND: Detected"
        else:
            status_color = (0, 0, 255)
            status_text = "HAND: None"
        cv2.putText(canvas, status_text, (x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y_pos += line_height + 15
        
        # Transformation values
        cv2.putText(canvas, "TRANSFORMS", (x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        y_pos += line_height
        
        # Scale
        scale_bar = int((gesture.scale_factor - 0.5) / 2.0 * 100)
        scale_bar = max(0, min(100, scale_bar))
        cv2.putText(canvas, f"Scale: {gesture.scale_factor:.2f}x", (x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20
        cv2.rectangle(canvas, (x + 10, y_pos - 10), (x + 10 + 100, y_pos), (60, 60, 60), -1)
        cv2.rectangle(canvas, (x + 10, y_pos - 10), (x + 10 + scale_bar, y_pos), (0, 200, 200), -1)
        y_pos += line_height
        
        # Rotation
        cv2.putText(canvas, f"Rotation: {gesture.rotation_angle:.1f} deg", (x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += line_height
        
        # Translation
        tx, ty = gesture.translation
        cv2.putText(canvas, f"Pan X: {tx}", (x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 22
        cv2.putText(canvas, f"Pan Y: {ty}", (x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += line_height
        
        # Flip status
        flip_text = "Flip: ON" if gesture.flip_horizontal else "Flip: OFF"
        flip_color = (0, 255, 255) if gesture.flip_horizontal else (100, 100, 100)
        cv2.putText(canvas, flip_text, (x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, flip_color, 1)
        y_pos += line_height + 10
        
        # Current image
        if self.available_images and self.current_image_index < len(self.available_images):
            _, img_name = self.available_images[self.current_image_index]
            cv2.putText(canvas, f"Image: {img_name}", (x + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
            y_pos += line_height
        
        # Controls hint
        cv2.putText(canvas, "CONTROLS", (x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        y_pos += line_height
        
        controls = [
            "q: Quit",
            "r: Reset",
            "c: Calibrate",
            "s: Screenshot",
            "1-4: Switch img"
        ]
        
        for ctrl in controls:
            cv2.putText(canvas, ctrl, (x + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            y_pos += 18
    
    def _handle_input(self):
        """Handle keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            self.running = False
        
        elif key == ord('r'):
            # Reset all
            self.recognizer.reset()
            self.transformer.reset()
            self.last_gesture = GestureResult()
            print("[RESET] All transformations reset")
        
        elif key == ord('c'):
            # Recalibrate
            self.recognizer.reset_calibration()
            print("[CALIBRATE] Gesture calibration reset")
        
        elif key == ord('s'):
            # Save screenshot
            filename = f"screenshot_{int(time.time())}.png"
            # Get current display
            ret, frame = self.webcam.read()
            if ret:
                frame = cv2.flip(frame, 1)
                landmarks = self.detector.detect(frame)
                gesture = self.last_gesture
                transformed = self.transformer.apply_all(
                    scale=gesture.scale_factor,
                    rotation=gesture.rotation_angle,
                    translation=gesture.translation,
                    flip_horizontal=gesture.flip_horizontal
                )
                self.detector.draw_landmarks(frame, landmarks)
                display = self._create_display(frame, transformed, gesture)
                cv2.imwrite(filename, display)
                print(f"[SAVED] {filename}")
        
        # Number keys 1-4 to switch images
        elif ord('1') <= key <= ord('4'):
            idx = key - ord('1')
            if idx < len(self.available_images):
                path, name = self.available_images[idx]
                self.transformer = ImageTransformer(
                    image_source=path,
                    output_size=(400, 400)
                )
                self.current_image_index = idx
                self.recognizer.reset()
                self.last_gesture = GestureResult()
                print(f"[IMAGE] Switched to: {name}")
    
    def _cleanup(self):
        """Clean up resources."""
        print()
        print("Shutting down...")
        self.webcam.release()
        self.detector.release()
        cv2.destroyAllWindows()
        print("Done!")


def main():
    """Entry point."""
    print()
    print("=" * 60)
    print("  GESTURE-CONTROLLED IMAGE MANIPULATOR")
    print("  Real-Time Hand Gesture Based Image Transformations")
    print("  Using OpenCV and MediaPipe")
    print("=" * 60)
    print()
    
    # Check for custom image
    image_path = None
    
    # Look for sample image (prefer 3D scene)
    possible_paths = [
        "assets/3d_scene.png",
        "assets/sample_image.png",
        "assets/3d_cube.png",
        "assets/3d_sphere.png",
        "assets/3d_pyramid.png",
        "assets/sample_image.jpg",
        "../assets/sample_image.png",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            image_path = path
            print(f"Using image: {path}")
            break
    
    if image_path is None:
        print("No sample image found - using generated test image")
        print("Run 'python src/generate_3d_image.py' to create 3D images")
    
    print()
    
    try:
        app = GestureControlApp(image_path=image_path)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
=======
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
>>>>>>> 1f46c2d5f244e891af626f383ac337fdf651807d
