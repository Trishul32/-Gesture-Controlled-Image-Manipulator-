"""
Demo script for Person A - Hand Detection Module
Tests the HandDetector independently before integration.

Run: python src/demo_hand_detection.py

Controls:
    q - Quit
    r - Reset (no action needed, just clears any state)
    s - Save screenshot
"""

import cv2
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hand_detector import HandDetector, WebcamCapture


def draw_info_panel(frame, info_lines, color=(0, 255, 0)):
    """Draw info panel with text lines."""
    # Background box
    box_height = 30 + len(info_lines) * 25
    cv2.rectangle(frame, (5, 5), (320, box_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (320, box_height), color, 2)
    
    # Text
    for i, text in enumerate(info_lines):
        cv2.putText(
            frame, text, (15, 30 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )


def draw_landmark_debug(frame, landmarks, w, h):
    """Draw additional debug info for key landmarks."""
    if not landmarks:
        return
    
    # Draw line between thumb and index (pinch gesture indicator)
    thumb = landmarks[HandDetector.THUMB_TIP]
    index = landmarks[HandDetector.INDEX_TIP]
    
    thumb_px = thumb.to_pixel(w, h)
    index_px = index.to_pixel(w, h)
    
    # Yellow line for pinch distance
    cv2.line(frame, thumb_px, index_px, (0, 255, 255), 2)
    
    # Draw circles on fingertips
    tips = [4, 8, 12, 16, 20]  # All fingertip indices
    colors = [
        (0, 0, 255),    # Red - Thumb
        (0, 255, 0),    # Green - Index
        (255, 0, 0),    # Blue - Middle
        (255, 255, 0),  # Cyan - Ring
        (255, 0, 255),  # Magenta - Pinky
    ]
    
    for tip_idx, color in zip(tips, colors):
        px = landmarks[tip_idx].to_pixel(w, h)
        cv2.circle(frame, px, 8, color, -1)
        cv2.circle(frame, px, 10, (255, 255, 255), 2)
    
    # Draw wrist
    wrist_px = landmarks[HandDetector.WRIST].to_pixel(w, h)
    cv2.circle(frame, wrist_px, 10, (0, 165, 255), -1)  # Orange


def main():
    print("=" * 60)
    print("  Hand Detection Demo - Person A")
    print("  Gesture-Controlled Image Manipulator Project")
    print("=" * 60)
    print()
    print("Controls:")
    print("  q - Quit")
    print("  s - Save screenshot")
    print()
    
    # Initialize
    try:
        detector = HandDetector(
            max_hands=1,
            detection_confidence=0.7,
            tracking_confidence=0.7
        )
        print("[OK] HandDetector initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize HandDetector: {e}")
        return
    
    try:
        cap = WebcamCapture(camera_id=0, width=640, height=480)
        print(f"[OK] Webcam opened ({cap.width}x{cap.height})")
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        detector.release()
        return
    
    print()
    print("Show your hand to the camera...")
    print()
    
    # FPS calculation
    fps_start = time.time()
    frame_count = 0
    fps = 0
    
    # Detection stats
    frames_with_hand = 0
    total_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        total_frames += 1
        
        # Detect hand
        landmarks = detector.detect(frame)
        
        # Draw MediaPipe skeleton
        detector.draw_landmarks(frame, landmarks)
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()
        
        # Prepare info display
        if landmarks:
            frames_with_hand += 1
            detection_rate = (frames_with_hand / total_frames) * 100
            
            thumb = landmarks[HandDetector.THUMB_TIP]
            index = landmarks[HandDetector.INDEX_TIP]
            wrist = landmarks[HandDetector.WRIST]
            
            # Calculate pinch distance (for Person B reference)
            import math
            pinch_dist = math.sqrt(
                (thumb.x - index.x)**2 + (thumb.y - index.y)**2
            )
            
            info = [
                f"FPS: {fps:.1f}",
                f"Status: HAND DETECTED",
                f"Landmarks: 21/21",
                f"Detection rate: {detection_rate:.1f}%",
                f"",
                f"Thumb: ({thumb.x:.2f}, {thumb.y:.2f})",
                f"Index: ({index.x:.2f}, {index.y:.2f})",
                f"Wrist: ({wrist.x:.2f}, {wrist.y:.2f})",
                f"Pinch dist: {pinch_dist:.3f}",
            ]
            color = (0, 255, 0)  # Green
            
            # Draw extra debug visualization
            draw_landmark_debug(frame, landmarks, w, h)
        else:
            info = [
                f"FPS: {fps:.1f}",
                f"Status: NO HAND",
                f"",
                f"Show your hand to camera",
            ]
            color = (0, 0, 255)  # Red
        
        draw_info_panel(frame, info, color)
        
        # Instructions at bottom
        cv2.putText(
            frame, "Press 'q' to quit | 's' to screenshot", 
            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        
        # Display
        cv2.imshow("Hand Detection - Person A", frame)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[SAVED] {filename}")
    
    # Cleanup
    cap.release()
    detector.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print()
    print("=" * 60)
    print("  Session Summary")
    print("=" * 60)
    print(f"  Total frames: {total_frames}")
    print(f"  Frames with hand: {frames_with_hand}")
    if total_frames > 0:
        print(f"  Detection rate: {(frames_with_hand/total_frames)*100:.1f}%")
    print(f"  Final FPS: {fps:.1f}")
    print()
    print("Demo complete!")


if __name__ == "__main__":
    main()
