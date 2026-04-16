"""
Image Transformation Module - Person C
Applies geometric transformations to images using OpenCV.

Transformations Supported:
    - Scaling (zoom in/out) using cv2.resize()
    - Rotation using cv2.getRotationMatrix2D() + cv2.warpAffine()
    - Translation (panning) using affine transformation
    - Reflection (flip) using cv2.flip()

Usage:
    from image_transformer import ImageTransformer
    
    transformer = ImageTransformer("sample_image.jpg")
    result = transformer.apply_all(scale=1.5, rotation=45, translation=(10, 20), flip=True)
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import os


class ImageTransformer:
    """
    Applies geometric transformations to an image.
    
    Transformation Matrices:
        Scaling: S = [[sx, 0], [0, sy]]
        Rotation: R = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        Translation: T = [[1, 0, tx], [0, 1, ty]]
        Reflection: Handled by cv2.flip()
    
    Example:
        transformer = ImageTransformer("my_image.jpg")
        
        # Apply individual transformations
        scaled = transformer.apply_scale(1.5)
        rotated = transformer.apply_rotation(45)
        
        # Or apply all at once
        result = transformer.apply_all(
            scale=1.2,
            rotation=30,
            translation=(50, 25),
            flip_horizontal=True
        )
        
        cv2.imshow("Result", result)
    """
    
    def __init__(
        self, 
        image_source: str = None,
        output_size: Tuple[int, int] = (400, 400)
    ):
        """
        Initialize image transformer.
        
        Args:
            image_source: Path to image file, or None to use a generated image
            output_size: (width, height) of output display
        """
        self.output_size = output_size
        
        if image_source and os.path.exists(image_source):
            self.original = cv2.imread(image_source)
            if self.original is None:
                raise ValueError(f"Could not load image: {image_source}")
        else:
            # Generate a colorful test image
            self.original = self._generate_test_image()
        
        # Store original dimensions
        self.orig_height, self.orig_width = self.original.shape[:2]
        
        # Current transformation state
        self.current = self.original.copy()
    
    def _generate_test_image(self) -> np.ndarray:
        """Generate a colorful test image with shapes and text."""
        size = 400
        img = np.ones((size, size, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Draw colorful shapes
        # Red circle
        cv2.circle(img, (100, 100), 60, (0, 0, 255), -1)
        cv2.circle(img, (100, 100), 60, (0, 0, 180), 3)
        
        # Green rectangle
        cv2.rectangle(img, (220, 50), (350, 150), (0, 200, 0), -1)
        cv2.rectangle(img, (220, 50), (350, 150), (0, 150, 0), 3)
        
        # Blue triangle
        pts = np.array([[200, 350], (100, 200), (300, 200)], np.int32)
        cv2.fillPoly(img, [pts], (255, 100, 0))
        cv2.polylines(img, [pts], True, (200, 80, 0), 3)
        
        # Yellow star shape (pentagon)
        center = (300, 300)
        pts = []
        for i in range(5):
            angle = i * 72 - 90  # Start from top
            r = 50 if i % 2 == 0 else 25
            x = int(center[0] + r * np.cos(np.radians(angle)))
            y = int(center[1] + r * np.sin(np.radians(angle)))
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], (0, 255, 255))
        
        # Add text
        cv2.putText(img, "GESTURE", (80, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 3)
        cv2.putText(img, "CONTROL", (90, 270),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 3)
        
        # Add border
        cv2.rectangle(img, (5, 5), (size-5, size-5), (100, 100, 100), 2)
        
        # Add corner markers (useful for seeing rotation)
        marker_size = 20
        cv2.rectangle(img, (10, 10), (10 + marker_size, 10 + marker_size), (255, 0, 0), -1)
        cv2.rectangle(img, (size - 30, 10), (size - 10, 10 + marker_size), (0, 255, 0), -1)
        cv2.rectangle(img, (10, size - 30), (10 + marker_size, size - 10), (0, 0, 255), -1)
        cv2.rectangle(img, (size - 30, size - 30), (size - 10, size - 10), (255, 255, 0), -1)
        
        return img
    
    def apply_scale(self, scale_factor: float) -> np.ndarray:
        """
        Scale the image around its center (zoom effect).
        
        Uses cv2.warpAffine() with a scale transformation matrix centered
        on the image. This zooms into the object without changing canvas size.
        
        Args:
            scale_factor: Scaling factor (1.0 = original, >1.0 = zoom in, <1.0 = zoom out)
            
        Returns:
            Scaled image with same dimensions as original
        """
        if scale_factor <= 0:
            scale_factor = 0.1
        
        h, w = self.orig_height, self.orig_width
        center = (w / 2, h / 2)
        
        # Create scale transformation matrix:
        # Scale around center point
        # M = [[scale, 0, cx*(1-scale)], [0, scale, cy*(1-scale)]]
        scale_matrix = cv2.getRotationMatrix2D(center, 0, scale_factor)
        
        # Apply scaling with warpAffine (maintains canvas size)
        scaled = cv2.warpAffine(
            self.original,
            scale_matrix,
            (w, h),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_LINEAR
        )
        
        return scaled
    
    def apply_rotation(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate the image around its center.
        
        Uses cv2.getRotationMatrix2D() and cv2.warpAffine().
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counter-clockwise)
            
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def apply_translation(
        self, 
        image: np.ndarray, 
        tx: int, 
        ty: int
    ) -> np.ndarray:
        """
        Translate (shift) the image.
        
        Uses affine transformation matrix [[1, 0, tx], [0, 1, ty]].
        
        Args:
            image: Input image
            tx: Horizontal shift in pixels (positive = right)
            ty: Vertical shift in pixels (positive = down)
            
        Returns:
            Translated image
        """
        h, w = image.shape[:2]
        
        # Translation matrix
        translation_matrix = np.float32([
            [1, 0, tx],
            [0, 1, ty]
        ])
        
        # Apply translation
        translated = cv2.warpAffine(
            image, 
            translation_matrix, 
            (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return translated
    
    def apply_reflection(
        self, 
        image: np.ndarray, 
        horizontal: bool = True
    ) -> np.ndarray:
        """
        Flip the image horizontally or vertically.
        
        Uses cv2.flip().
        
        Args:
            image: Input image
            horizontal: If True, flip horizontally. If False, flip vertically.
            
        Returns:
            Flipped image
        """
        flip_code = 1 if horizontal else 0
        return cv2.flip(image, flip_code)
    
    def apply_all(
        self,
        scale: float = 1.0,
        rotation: float = 0.0,
        translation: Tuple[int, int] = (0, 0),
        flip_horizontal: bool = False
    ) -> np.ndarray:
        """
        Apply all transformations: combined scale+rotate+translate → flip.
        
        Uses a single combined transformation matrix for scale, rotation, and
        translation to minimize quality loss and ensure center-based zoom.
        
        Args:
            scale: Scale factor (1.0 = no change, >1.0 = zoom in, <1.0 = zoom out)
            rotation: Rotation angle in degrees
            translation: (tx, ty) translation in pixels
            flip_horizontal: Whether to flip horizontally
            
        Returns:
            Transformed image resized to output_size
        """
        h, w = self.orig_height, self.orig_width
        center = (w / 2, h / 2)
        
        # Build combined transformation matrix:
        # M combines: Scale around center + Rotation around center + Translation
        
        # Get rotation+scale matrix from OpenCV (includes both in one call)
        # This scales and rotates around the center point
        combined_matrix = cv2.getRotationMatrix2D(center, rotation, scale)
        
        # Add translation (modify the translation component of the matrix)
        tx, ty = translation
        combined_matrix[0, 2] += tx
        combined_matrix[1, 2] += ty
        
        # Apply combined transformation in one warpAffine call
        result = cv2.warpAffine(
            self.original,
            combined_matrix,
            (w, h),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_LINEAR
        )
        
        # Apply reflection (separate step - cannot be combined with affine)
        if flip_horizontal:
            result = self.apply_reflection(result, horizontal=True)
        
        # Store current state
        self.current = result
        
        # Resize to output size for display
        display = self._fit_to_output(result)
        
        return display
    
    def _fit_to_output(self, image: np.ndarray) -> np.ndarray:
        """
        Fit image to output size while maintaining aspect ratio.
        Centers the image on a gray background if needed.
        """
        h, w = image.shape[:2]
        out_w, out_h = self.output_size
        
        # Calculate scaling to fit
        scale_w = out_w / w
        scale_h = out_h / h
        scale = min(scale_w, scale_h)
        
        # Resize image
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create output canvas (gray background)
        output = np.ones((out_h, out_w, 3), dtype=np.uint8) * 50
        
        # Center the image
        x_offset = (out_w - new_w) // 2
        y_offset = (out_h - new_h) // 2
        
        output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return output
    
    def get_original(self) -> np.ndarray:
        """Get the original image."""
        return self.original.copy()
    
    def reset(self) -> None:
        """Reset to original image."""
        self.current = self.original.copy()
    
    def set_image(self, image: np.ndarray) -> None:
        """Set a new source image."""
        self.original = image.copy()
        self.orig_height, self.orig_width = self.original.shape[:2]
        self.current = self.original.copy()


# Quick test
if __name__ == "__main__":
    import time
    
    print("Testing ImageTransformer...")
    
    # Create transformer with test image
    transformer = ImageTransformer()
    
    # Test individual transformations
    print("Testing scale...")
    scaled = transformer.apply_scale(1.5)
    cv2.imshow("Scaled 1.5x", scaled)
    
    print("Testing rotation...")
    rotated = transformer.apply_rotation(transformer.original, 45)
    cv2.imshow("Rotated 45°", rotated)
    
    print("Testing translation...")
    translated = transformer.apply_translation(transformer.original, 50, 30)
    cv2.imshow("Translated", translated)
    
    print("Testing flip...")
    flipped = transformer.apply_reflection(transformer.original, horizontal=True)
    cv2.imshow("Flipped", flipped)
    
    print("Testing combined...")
    combined = transformer.apply_all(
        scale=1.2,
        rotation=30,
        translation=(20, 10),
        flip_horizontal=False
    )
    cv2.imshow("Combined", combined)
    
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Test complete!")