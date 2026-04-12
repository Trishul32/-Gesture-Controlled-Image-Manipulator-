import math
from typing import List, Optional
from hand_detector import Landmark, HandDetector


class GestureRecognizer:
    def __init__(self):
        self.prev_centroid = None
        self.prev_angle = None

    # -------------------------
    # Utility Functions
    # -------------------------

    def _distance(self, p1: Landmark, p2: Landmark) -> float:
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _angle(self, p1: Landmark, p2: Landmark) -> float:
        return math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x))

    def _centroid(self, landmarks: List[Landmark]):
        x = sum([lm.x for lm in landmarks]) / len(landmarks)
        y = sum([lm.y for lm in landmarks]) / len(landmarks)
        return (x, y)

    # -------------------------
    # Gesture Detection
    # -------------------------

    def detect_pinch(self, landmarks):
        thumb = landmarks[HandDetector.THUMB_TIP]
        index = landmarks[HandDetector.INDEX_TIP]

        dist = self._distance(thumb, index)

        if dist < 0.05:
            return "PINCH"
        return None

    def detect_rotation(self, landmarks):
        wrist = landmarks[HandDetector.WRIST]
        index = landmarks[HandDetector.INDEX_TIP]

        angle = self._angle(wrist, index)

        if self.prev_angle is not None:
            diff = angle - self.prev_angle

            if abs(diff) > 10:
                self.prev_angle = angle
                return "ROTATE", diff

        self.prev_angle = angle
        return None

    def detect_swipe(self, landmarks):
        centroid = self._centroid(landmarks)

        if self.prev_centroid is not None:
            dx = centroid[0] - self.prev_centroid[0]

            if abs(dx) > 0.1:
                self.prev_centroid = centroid
                return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"

        self.prev_centroid = centroid
        return None

    def detect_open_palm(self, landmarks):
        tips = [8, 12, 16, 20]  # Index to pinky tips
        base = landmarks[HandDetector.WRIST]

        count = 0
        for tip in tips:
            if landmarks[tip].y < base.y:
                count += 1

        if count >= 4:
            return "OPEN_PALM"
        return None

    # -------------------------
    # Main Interface
    # -------------------------

    def recognize(self, landmarks: Optional[List[Landmark]]):
        if not landmarks:
            return None

        # Priority order
        if self.detect_pinch(landmarks):
            return "PINCH"

        rot = self.detect_rotation(landmarks)
        if rot:
            return rot

        swipe = self.detect_swipe(landmarks)
        if swipe:
            return swipe

        if self.detect_open_palm(landmarks):
            return "OPEN_PALM"

        return "NONE"