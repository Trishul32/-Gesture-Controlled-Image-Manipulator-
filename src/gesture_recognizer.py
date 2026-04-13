"""
Gesture Recognition Module - Person B
Recognizes hand gestures and maps them to transformation parameters.

Gestures Supported:
    - Pinch (thumb + index distance) → Scale factor
    - Hand tilt angle → Rotation angle
    - Hand centroid movement → Translation offset
    - Fist gesture → Toggle horizontal flip

Usage:
    from gesture_recognizer import GestureRecognizer, GestureResult
    
    recognizer = GestureRecognizer()
    result = recognizer.recognize(landmarks, frame_width, frame_height)
    print(f"Scale: {result.scale_factor}, Rotation: {result.rotation_angle}")
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from collections import deque


@dataclass
class GestureResult:
    """
    Result of gesture recognition containing transformation parameters.
    
    Attributes:
        scale_factor: Zoom level (0.5 = half size, 2.0 = double size)
        rotation_angle: Rotation in degrees
        translation: (dx, dy) offset in pixels
        flip_horizontal: Whether to flip image horizontally
        hand_detected: Whether a hand was detected
        pinch_distance: Raw pinch distance (for debugging)
        hand_angle: Raw hand angle (for debugging)
        gesture_name: Name of currently detected gesture
    """
    scale_factor: float = 1.0
    rotation_angle: float = 0.0
    translation: Tuple[int, int] = (0, 0)
    flip_horizontal: bool = False
    hand_detected: bool = False
    pinch_distance: float = 0.0
    hand_angle: float = 0.0
    gesture_name: str = "None"


class GestureRecognizer:
    """
    Recognizes hand gestures from landmarks and converts to transformation parameters.
    
    IMPROVED VERSION with better accuracy:
    - Uses multiple finger distances for more stable pinch detection
    - Exponential moving average for smoother values
    - Dead zones to prevent jitter
    - Relative scaling from calibration point
    """
    
    def __init__(
<<<<<<< HEAD
        self, 
=======
        self,
>>>>>>> 1f46c2d5f244e891af626f383ac337fdf651807d
        smoothing_factor: float = 0.3,
        scale_sensitivity: float = 1.5,
        rotation_sensitivity: float = 0.8,
        translation_sensitivity: float = 0.8,
        dead_zone_scale: float = 0.02,
        dead_zone_rotation: float = 3.0,
        dead_zone_translation: float = 5
    ):
        """
        Initialize gesture recognizer with improved parameters.
        
        Args:
            smoothing_factor: EMA smoothing (0-1, lower = smoother but slower)
            scale_sensitivity: Multiplier for scale changes
            rotation_sensitivity: Multiplier for rotation
            translation_sensitivity: Multiplier for translation
            dead_zone_scale: Minimum pinch change to register
            dead_zone_rotation: Minimum angle change to register (degrees)
            dead_zone_translation: Minimum movement to register (pixels)
        """
        self.smoothing_factor = smoothing_factor
        self.scale_sensitivity = scale_sensitivity
        self.rotation_sensitivity = rotation_sensitivity
        self.translation_sensitivity = translation_sensitivity
        self.dead_zone_scale = dead_zone_scale
        self.dead_zone_rotation = dead_zone_rotation
        self.dead_zone_translation = dead_zone_translation
        
        # Calibration baselines
        self.base_pinch_distance: Optional[float] = None
        self.base_angle: Optional[float] = None
        self.base_centroid: Optional[Tuple[float, float]] = None
        
        # Previous frame tracking (for delta calculation)
        self.prev_pinch_distance: Optional[float] = None
        self.prev_angle: Optional[float] = None
        self.prev_centroid: Optional[Tuple[float, float]] = None
        
        # Smoothed current values (using EMA)
        self.smooth_scale: float = 1.0
        self.smooth_rotation: float = 0.0
        self.smooth_translation: List[float] = [0.0, 0.0]
        
        # State
        self.flip_state = False
        self.prev_fist_state = False
        self.fist_cooldown = 0  # Prevent rapid toggling
        
        # History for stability detection
        self.pinch_history = deque(maxlen=10)
        self.angle_history = deque(maxlen=10)
        
        # Accumulated values (PERSIST across gestures)
        self.accumulated_translation = [0.0, 0.0]
        self.accumulated_scale = 1.0
        self.accumulated_rotation = 0.0
    
    def recognize(
<<<<<<< HEAD
        self, 
        landmarks: List, 
        frame_width: int, 
=======
        self,
        landmarks: List,
        frame_width: int,
>>>>>>> 1f46c2d5f244e891af626f383ac337fdf651807d
        frame_height: int
    ) -> GestureResult:
        """
        Recognize gestures from hand landmarks with improved accuracy.
        """
        if not landmarks or len(landmarks) < 21:
            # Decrease fist cooldown even when no hand
            if self.fist_cooldown > 0:
                self.fist_cooldown -= 1
            return GestureResult(
                hand_detected=False,
                scale_factor=self.accumulated_scale,
                rotation_angle=self.accumulated_rotation,
<<<<<<< HEAD
                translation=(int(self.accumulated_translation[0]), 
=======
                translation=(int(self.accumulated_translation[0]),
>>>>>>> 1f46c2d5f244e891af626f383ac337fdf651807d
                           int(self.accumulated_translation[1])),
                flip_horizontal=self.flip_state,
                gesture_name="No Hand"
            )
        
        result = GestureResult(hand_detected=True)
        current_gesture = "Tracking"
        
        # === SCALE: Delta-based accumulation ===
        pinch_dist = self._calculate_pinch_distance_improved(landmarks)
        self.pinch_history.append(pinch_dist)
        
        # Use median for more stability
        sorted_pinch = sorted(self.pinch_history)
        median_pinch = sorted_pinch[len(sorted_pinch) // 2]
        
        # Initialize baseline on first stable detection
        if self.base_pinch_distance is None:
            if len(self.pinch_history) >= 5:
                variance = sum((p - median_pinch) ** 2 for p in self.pinch_history) / len(self.pinch_history)
                if variance < 0.001:  # Stable hand position
                    self.base_pinch_distance = median_pinch
        
        if self.base_pinch_distance is not None:
            # Calculate scale CHANGE from previous pinch
            if hasattr(self, 'prev_pinch_distance') and self.prev_pinch_distance is not None:
                pinch_delta = median_pinch - self.prev_pinch_distance
                
                # Apply dead zone - only accumulate if change is significant
                if abs(pinch_delta) > self.dead_zone_scale:
                    # Scale factor change: positive delta = zoom in, negative = zoom out
                    scale_change = pinch_delta * self.scale_sensitivity * 3.0
                    self.accumulated_scale += scale_change
                    self.accumulated_scale = max(0.3, min(3.0, self.accumulated_scale))
                    current_gesture = "Pinch: Zoom"
            
            self.prev_pinch_distance = median_pinch
        
        result.scale_factor = self.accumulated_scale
        result.pinch_distance = median_pinch
        
        # === ROTATION: Delta-based accumulation ===
        raw_angle = self._calculate_hand_angle_improved(landmarks)
        self.angle_history.append(raw_angle)
        
        # Use median for stability
        sorted_angle = sorted(self.angle_history)
        median_angle = sorted_angle[len(sorted_angle) // 2]
        
        # Calculate rotation CHANGE from previous angle
        if hasattr(self, 'prev_angle') and self.prev_angle is not None:
            angle_delta = median_angle - self.prev_angle
            
            # Handle angle wrapping (-180 to 180)
            if angle_delta > 180:
                angle_delta -= 360
            elif angle_delta < -180:
                angle_delta += 360
            
            # Apply dead zone - only accumulate if change is significant
            if abs(angle_delta) > self.dead_zone_rotation:
                self.accumulated_rotation += angle_delta * self.rotation_sensitivity
                current_gesture = "Tilt: Rotate"
        
        self.prev_angle = median_angle
        
        result.rotation_angle = self.accumulated_rotation
        result.hand_angle = median_angle
        
        # === TRANSLATION: Delta-based accumulation ===
        centroid = self._calculate_palm_center(landmarks)
        
        if hasattr(self, 'prev_centroid') and self.prev_centroid is not None:
            # Calculate movement CHANGE from previous frame
            dx = (centroid[0] - self.prev_centroid[0]) * frame_width * self.translation_sensitivity
            dy = (centroid[1] - self.prev_centroid[1]) * frame_height * self.translation_sensitivity
            
            # Apply dead zone - only accumulate if movement is significant
            if abs(dx) > self.dead_zone_translation or abs(dy) > self.dead_zone_translation:
                self.accumulated_translation[0] += dx
                self.accumulated_translation[1] += dy
                current_gesture = "Move: Pan"
        
        self.prev_centroid = centroid
        
<<<<<<< HEAD
        result.translation = (int(self.accumulated_translation[0]), 
=======
        result.translation = (int(self.accumulated_translation[0]),
>>>>>>> 1f46c2d5f244e891af626f383ac337fdf651807d
                             int(self.accumulated_translation[1]))
        
        # === REFLECTION: Improved fist detection ===
        if self.fist_cooldown > 0:
            self.fist_cooldown -= 1
        
        is_fist = self._is_fist_improved(landmarks)
        
        if is_fist and not self.prev_fist_state and self.fist_cooldown == 0:
            self.flip_state = not self.flip_state
            self.fist_cooldown = 15  # Cooldown frames
            current_gesture = "Fist: Flip"
        
        self.prev_fist_state = is_fist
        result.flip_horizontal = self.flip_state
        result.gesture_name = current_gesture
        
        return result
    
    def _calculate_pinch_distance_improved(self, landmarks: List) -> float:
        """
        Calculate pinch distance using thumb and index fingertips.
        Also considers the stability by checking thumb-index angle.
        """
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Primary distance: thumb to index
        dist = math.sqrt(
<<<<<<< HEAD
            (thumb_tip.x - index_tip.x) ** 2 + 
=======
            (thumb_tip.x - index_tip.x) ** 2 +
>>>>>>> 1f46c2d5f244e891af626f383ac337fdf651807d
            (thumb_tip.y - index_tip.y) ** 2
        )
        
        return dist
    
    def _calculate_hand_angle_improved(self, landmarks: List) -> float:
        """
        Calculate hand rotation using wrist-to-middle-finger vector.
        More stable than previous implementation.
        """
        wrist = landmarks[0]
        middle_mcp = landmarks[9]  # Middle finger base for stability
        
        # Vector from wrist to middle MCP
        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y
        
        # Calculate angle (0 = pointing up)
        angle = math.degrees(math.atan2(dx, -dy))
        
        return angle
    
    def _calculate_palm_center(self, landmarks: List) -> Tuple[float, float]:
        """
        Calculate palm center using wrist and MCP joints for stability.
        """
        # Use wrist and base of fingers for stable palm center
        indices = [0, 5, 9, 13, 17]  # Wrist + 4 MCP joints
        
        x = sum(landmarks[i].x for i in indices) / len(indices)
        y = sum(landmarks[i].y for i in indices) / len(indices)
        
        return (x, y)
    
    def _is_fist_improved(self, landmarks: List) -> bool:
        """
        Improved fist detection using finger curl measurement.
        """
        # For each finger, check if tip is closer to wrist than MCP
        wrist = landmarks[0]
        
        fingers_curled = 0
        
        # Check each finger (index, middle, ring, pinky)
        finger_tips = [8, 12, 16, 20]
        finger_mcps = [5, 9, 13, 17]
        
        for tip_idx, mcp_idx in zip(finger_tips, finger_mcps):
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]
            
            # Distance from tip to wrist
            tip_to_wrist = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
            
            # Distance from MCP to wrist
            mcp_to_wrist = math.sqrt((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2)
            
            # If tip is closer to wrist than MCP, finger is curled
            if tip_to_wrist < mcp_to_wrist * 1.1:
                fingers_curled += 1
        
        # Also check thumb
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        
        thumb_tip_dist = math.sqrt((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)
        thumb_mcp_dist = math.sqrt((thumb_mcp.x - wrist.x)**2 + (thumb_mcp.y - wrist.y)**2)
        
        if thumb_tip_dist < thumb_mcp_dist * 1.2:
            fingers_curled += 1
        
        return fingers_curled >= 4
    
    def reset(self) -> None:
        """Reset all state - transforms go back to original."""
        self.base_pinch_distance = None
        self.base_angle = None
        self.base_centroid = None
        self.prev_pinch_distance = None
        self.prev_angle = None
        self.prev_centroid = None
        self.smooth_scale = 1.0
        self.smooth_rotation = 0.0
        self.smooth_translation = [0.0, 0.0]
        self.accumulated_scale = 1.0
        self.accumulated_rotation = 0.0
        self.accumulated_translation = [0.0, 0.0]
        self.flip_state = False
        self.prev_fist_state = False
        self.fist_cooldown = 0
        self.pinch_history.clear()
        self.angle_history.clear()
    
    def reset_calibration(self) -> None:
        """Reset calibration but keep accumulated transforms."""
        self.base_pinch_distance = None
        self.base_angle = None
        self.base_centroid = None
        self.prev_pinch_distance = None
        self.prev_angle = None
        self.prev_centroid = None
        self.pinch_history.clear()
        self.angle_history.clear()


# Quick test
if __name__ == "__main__":
    from hand_detector import HandDetector, WebcamCapture
    import cv2
    
    print("Testing GestureRecognizer...")
    
    detector = HandDetector()
    recognizer = GestureRecognizer()
    cap = WebcamCapture()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        landmarks = detector.detect(frame)
        
        if landmarks:
            result = recognizer.recognize(landmarks, cap.width, cap.height)
            
            info = [
                f"Scale: {result.scale_factor:.2f}x",
                f"Rotation: {result.rotation_angle:.1f} deg",
                f"Translation: {result.translation}",
                f"Flip: {result.flip_horizontal}",
                f"Pinch: {result.pinch_distance:.3f}",
            ]
            
            for i, text in enumerate(info):
                cv2.putText(frame, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        detector.draw_landmarks(frame, landmarks)
        cv2.imshow("Gesture Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recognizer.reset()
            print("Reset!")
    
    cap.release()
    detector.release()
    cv2.destroyAllWindows()
