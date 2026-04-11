"""
Hand Detection Module - Person A
Detects hand landmarks using MediaPipe Tasks API (v0.10+).

Usage:
    from hand_detector import HandDetector, Landmark
    
    detector = HandDetector()
    landmarks = detector.detect(frame)  # Returns List[Landmark] or None
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os


@dataclass
class Landmark:
    """
    Single hand landmark with normalized coordinates.
    
    Attributes:
        x: Horizontal position (0.0 = left edge, 1.0 = right edge)
        y: Vertical position (0.0 = top edge, 1.0 = bottom edge)
        z: Depth relative to wrist (smaller = closer to camera)
    """
    x: float
    y: float
    z: float
    
    def to_pixel(self, frame_width: int, frame_height: int) -> Tuple[int, int]:
        """
        Convert normalized coordinates to pixel coordinates.
        
        Args:
            frame_width: Width of the frame in pixels
            frame_height: Height of the frame in pixels
            
        Returns:
            Tuple of (x, y) pixel coordinates
        """
        return (int(self.x * frame_width), int(self.y * frame_height))


class HandDetector:
    """
    Detects hand landmarks using MediaPipe Tasks API.
    
    Provides 21 hand landmarks per detected hand. Key landmark indices:
        0:  Wrist
        1:  Thumb CMC
        2:  Thumb MCP
        3:  Thumb IP
        4:  Thumb Tip
        5:  Index MCP
        6:  Index PIP
        7:  Index DIP
        8:  Index Tip
        9:  Middle MCP
        10: Middle PIP
        11: Middle DIP
        12: Middle Tip
        13: Ring MCP
        14: Ring PIP
        15: Ring DIP
        16: Ring Tip
        17: Pinky MCP
        18: Pinky PIP
        19: Pinky DIP
        20: Pinky Tip
    
    Example:
        detector = HandDetector()
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            landmarks = detector.detect(frame)
            
            if landmarks:
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                print(f"Thumb: ({thumb_tip.x:.2f}, {thumb_tip.y:.2f})")
            
            detector.draw_landmarks(frame, landmarks)
            cv2.imshow("Hand", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        detector.release()
        cap.release()
    """
    
    # Landmark index constants for easy reference
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    # Hand connections for drawing (pairs of landmark indices)
    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17)
    ]
    
    def __init__(
        self, 
        max_hands: int = 1, 
        detection_confidence: float = 0.7, 
        tracking_confidence: float = 0.7,
        model_path: str = None
    ):
        """
        Initialize hand detector.
        
        Args:
            max_hands: Maximum number of hands to detect (default: 1)
            detection_confidence: Minimum confidence for detection (0.0-1.0)
            tracking_confidence: Minimum confidence for tracking (0.0-1.0)
            model_path: Path to hand_landmarker.task model file
        """
        # Find model path
        if model_path is None:
            # Look in common locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'assets', 'hand_landmarker.task'),
                os.path.join(os.path.dirname(__file__), 'assets', 'hand_landmarker.task'),
                'assets/hand_landmarker.task',
                '../assets/hand_landmarker.task',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    "Could not find hand_landmarker.task model. "
                    "Download from: https://storage.googleapis.com/mediapipe-models/"
                    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                )
        
        # Configure HandLandmarker
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        self.landmarker = HandLandmarker.create_from_options(options)
        self._last_landmarks = None
        self._frame_shape = None
    
    def detect(self, frame) -> Optional[List[Landmark]]:
        """
        Detect hand landmarks in a frame.
        
        Args:
            frame: BGR image from cv2.VideoCapture
            
        Returns:
            List of 21 Landmark objects if hand detected, None otherwise.
            Landmarks are indexed 0-20 (see class docstring for mapping).
        """
        # Store frame shape
        self._frame_shape = frame.shape[:2]  # (height, width)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect landmarks
        result = self.landmarker.detect(mp_image)
        
        # Check if any hand was detected
        if not result.hand_landmarks:
            self._last_landmarks = None
            return None
        
        # Extract first hand's landmarks
        hand = result.hand_landmarks[0]
        landmarks = [
            Landmark(x=lm.x, y=lm.y, z=lm.z) 
            for lm in hand
        ]
        
        self._last_landmarks = landmarks
        return landmarks
    
    def draw_landmarks(self, frame, landmarks: Optional[List[Landmark]] = None) -> None:
        """
        Draw hand landmarks and connections on frame (modifies frame in-place).
        
        Args:
            frame: BGR image to draw on
            landmarks: List of Landmark objects (uses last detected if None)
        """
        if landmarks is None:
            landmarks = self._last_landmarks
        
        if landmarks is None:
            return
        
        h, w = frame.shape[:2]
        
        # Convert to pixel coordinates
        points = [lm.to_pixel(w, h) for lm in landmarks]
        
        # Draw connections
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            start = points[start_idx]
            end = points[end_idx]
            cv2.line(frame, start, end, (0, 255, 0), 2)
        
        # Draw landmarks
        for i, point in enumerate(points):
            # Fingertips in red, others in green
            if i in [4, 8, 12, 16, 20]:
                color = (0, 0, 255)  # Red for fingertips
                radius = 6
            elif i == 0:
                color = (255, 0, 0)  # Blue for wrist
                radius = 8
            else:
                color = (0, 255, 0)  # Green for others
                radius = 4
            
            cv2.circle(frame, point, radius, color, -1)
            cv2.circle(frame, point, radius + 2, (255, 255, 255), 1)
    
    def get_pixel_coords(
        self, 
        landmarks: List[Landmark], 
        frame_width: int, 
        frame_height: int
    ) -> List[Tuple[int, int]]:
        """
        Convert all landmarks to pixel coordinates.
        
        Args:
            landmarks: List of Landmark objects from detect()
            frame_width: Width of frame in pixels
            frame_height: Height of frame in pixels
            
        Returns:
            List of (x, y) pixel coordinate tuples
        """
        return [lm.to_pixel(frame_width, frame_height) for lm in landmarks]
    
    def is_hand_detected(self) -> bool:
        """Check if a hand was detected in the last frame."""
        return self._last_landmarks is not None
    
    def release(self) -> None:
        """Release MediaPipe resources. Call when done using detector."""
        self.landmarker.close()


class WebcamCapture:
    """
    Simple webcam capture wrapper.
    
    Example:
        cap = WebcamCapture()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    """
    
    def __init__(
        self, 
        camera_id: int = 0, 
        width: int = 640, 
        height: int = 480
    ):
        """
        Initialize webcam capture.
        
        Args:
            camera_id: Camera device ID (default: 0 for primary webcam)
            width: Desired frame width in pixels
            height: Desired frame height in pixels
            
        Raises:
            RuntimeError: If camera cannot be opened
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        # Get actual dimensions (may differ from requested)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def read(self) -> Tuple[bool, any]:
        """
        Read a frame from the webcam.
        
        Returns:
            Tuple of (success: bool, frame: ndarray or None)
        """
        return self.cap.read()
    
    def release(self) -> None:
        """Release the webcam resource."""
        self.cap.release()


# Quick test when run directly
if __name__ == "__main__":
    print("Testing HandDetector...")
    
    detector = HandDetector()
    cap = WebcamCapture()
    
    print(f"Webcam opened: {cap.width}x{cap.height}")
    print("Show your hand to the camera. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror
        landmarks = detector.detect(frame)
        
        if landmarks:
            print(f"Detected 21 landmarks! Thumb tip: ({landmarks[4].x:.2f}, {landmarks[4].y:.2f})")
        
        detector.draw_landmarks(frame, landmarks)
        cv2.imshow("Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    detector.release()
    cv2.destroyAllWindows()
    print("Test complete!")
