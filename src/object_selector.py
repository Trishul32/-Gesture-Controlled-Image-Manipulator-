"""
Object Selector Module
Allows selecting objects from images using various methods:
- Click-based selection with flood fill
- Contour detection
- Edge-based segmentation
- Color-based selection

Usage:
    from object_selector import ObjectSelector
    
    selector = ObjectSelector(image)
    mask = selector.select_at_point(x, y)
    selected_object = selector.extract_object(mask)
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class SelectedObject:
    """Represents a selected object from an image."""
    mask: np.ndarray           # Binary mask of the object
    contour: np.ndarray        # Contour points
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]    # Center point
    area: int                  # Pixel area
    image: np.ndarray          # Cropped image of object
    image_with_alpha: np.ndarray  # RGBA image with transparency


class ObjectSelector:
    """
    Selects objects from images using various segmentation methods.
    """
    
    def __init__(self, image: np.ndarray):
        """
        Initialize with an image.
        
        Args:
            image: BGR image (OpenCV format)
        """
        self.original = image.copy()
        self.image = image.copy()
        self.height, self.width = image.shape[:2]
        
        # Precompute useful representations
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.edges = cv2.Canny(self.gray, 50, 150)
        
        # Store detected contours
        self.contours = []
        self.selected_objects: List[SelectedObject] = []
        self._detect_contours()
    
    def _detect_contours(self):
        """Detect all contours in the image."""
        # Use adaptive thresholding for better edge detection
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        
        # Multiple thresholding approaches
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        min_area = (self.width * self.height) * 0.001  # At least 0.1% of image
        self.contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Sort by area (largest first)
        self.contours.sort(key=cv2.contourArea, reverse=True)
    
    def select_at_point(
        self, 
        x: int, 
        y: int, 
        tolerance: int = 20,
        method: str = "smart"
    ) -> Optional[SelectedObject]:
        """
        Select object at the given point.
        
        Args:
            x, y: Click coordinates
            tolerance: Color tolerance for flood fill
            method: Selection method - "smart", "flood", "contour", "grabcut"
            
        Returns:
            SelectedObject or None if nothing selected
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        
        if method == "smart":
            # Try contour first, fall back to flood fill
            obj = self._select_by_contour(x, y)
            if obj is None or obj.area < 100:
                obj = self._select_by_flood_fill(x, y, tolerance)
            return obj
        elif method == "flood":
            return self._select_by_flood_fill(x, y, tolerance)
        elif method == "contour":
            return self._select_by_contour(x, y)
        elif method == "grabcut":
            return self._select_by_grabcut(x, y)
        else:
            return self._select_by_flood_fill(x, y, tolerance)
    
    def _select_by_flood_fill(
        self, 
        x: int, 
        y: int, 
        tolerance: int = 20
    ) -> Optional[SelectedObject]:
        """Select object using flood fill from click point."""
        # Create mask for flood fill (needs to be 2 pixels larger)
        h, w = self.height, self.width
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Flood fill on a copy
        flood_image = self.image.copy()
        
        # Flood fill with a unique color
        cv2.floodFill(
            flood_image, 
            mask, 
            (x, y), 
            (255, 0, 255),  # Magenta fill color
            (tolerance, tolerance, tolerance),
            (tolerance, tolerance, tolerance),
            cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        )
        
        # Extract the filled region from mask
        object_mask = mask[1:-1, 1:-1]  # Remove padding
        
        return self._create_selected_object(object_mask)
    
    def _select_by_contour(self, x: int, y: int) -> Optional[SelectedObject]:
        """Select the contour containing the click point."""
        for contour in self.contours:
            # Check if point is inside contour
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                # Create mask from contour
                mask = np.zeros((self.height, self.width), np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                return self._create_selected_object(mask, contour)
        
        return None
    
    def _select_by_grabcut(
        self, 
        x: int, 
        y: int, 
        rect_size: int = 100
    ) -> Optional[SelectedObject]:
        """Select object using GrabCut algorithm."""
        # Define rectangle around click point
        x1 = max(0, x - rect_size)
        y1 = max(0, y - rect_size)
        x2 = min(self.width, x + rect_size)
        y2 = min(self.height, y + rect_size)
        
        rect = (x1, y1, x2 - x1, y2 - y1)
        
        # GrabCut requires mask initialization
        mask = np.zeros((self.height, self.width), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(
                self.image, mask, rect, 
                bgd_model, fgd_model, 
                5, cv2.GC_INIT_WITH_RECT
            )
            
            # Create binary mask from GrabCut result
            binary_mask = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 
                255, 0
            ).astype(np.uint8)
            
            return self._create_selected_object(binary_mask)
        except:
            return None
    
    def select_by_color(
        self, 
        x: int, 
        y: int, 
        tolerance: int = 30
    ) -> Optional[SelectedObject]:
        """Select all pixels of similar color."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        
        # Get color at click point (HSV for better color matching)
        target_hsv = self.hsv[y, x]
        
        # Create range for color matching
        lower = np.array([
            max(0, target_hsv[0] - tolerance),
            max(0, target_hsv[1] - tolerance * 2),
            max(0, target_hsv[2] - tolerance * 2)
        ])
        upper = np.array([
            min(179, target_hsv[0] + tolerance),
            min(255, target_hsv[1] + tolerance * 2),
            min(255, target_hsv[2] + tolerance * 2)
        ])
        
        # Create mask of matching colors
        mask = cv2.inRange(self.hsv, lower, upper)
        
        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return self._create_selected_object(mask)
    
    def _create_selected_object(
        self, 
        mask: np.ndarray, 
        contour: np.ndarray = None
    ) -> Optional[SelectedObject]:
        """Create SelectedObject from a binary mask."""
        # Find contours in mask if not provided
        if contour is None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        bbox = cv2.boundingRect(contour)
        x, y, w, h = bbox
        
        # Calculate center
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        # Calculate area
        area = cv2.contourArea(contour)
        
        if area < 10:  # Too small
            return None
        
        # Extract cropped image
        cropped = self.image[y:y+h, x:x+w].copy()
        
        # Create RGBA image with transparency
        cropped_mask = mask[y:y+h, x:x+w]
        rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = cropped_mask
        
        return SelectedObject(
            mask=mask,
            contour=contour,
            bbox=bbox,
            center=(cx, cy),
            area=int(area),
            image=cropped,
            image_with_alpha=rgba
        )
    
    def get_all_objects(self, min_area: int = 500) -> List[SelectedObject]:
        """Get all detected objects in the image."""
        objects = []
        
        for contour in self.contours:
            if cv2.contourArea(contour) < min_area:
                continue
            
            mask = np.zeros((self.height, self.width), np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            obj = self._create_selected_object(mask, contour)
            if obj:
                objects.append(obj)
        
        return objects
    
    def draw_objects_overlay(
        self, 
        objects: List[SelectedObject] = None,
        highlight_color: Tuple[int, int, int] = (0, 255, 0),
        show_bbox: bool = True,
        show_contour: bool = True
    ) -> np.ndarray:
        """Draw overlay showing detected/selected objects."""
        display = self.image.copy()
        
        if objects is None:
            objects = self.selected_objects
        
        for obj in objects:
            if show_contour:
                cv2.drawContours(display, [obj.contour], -1, highlight_color, 2)
            
            if show_bbox:
                x, y, w, h = obj.bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), highlight_color, 2)
                
                # Draw center point
                cv2.circle(display, obj.center, 5, (0, 0, 255), -1)
        
        return display
    
    def draw_all_contours(self) -> np.ndarray:
        """Draw all detected contours for visualization."""
        display = self.image.copy()
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 255), (255, 128, 0), (0, 128, 255)
        ]
        
        for i, contour in enumerate(self.contours):
            color = colors[i % len(colors)]
            cv2.drawContours(display, [contour], -1, color, 2)
        
        return display


class InteractiveObjectSelector:
    """
    Interactive object selection with mouse clicks.
    """
    
    def __init__(self, image: np.ndarray, window_name: str = "Object Selector"):
        """Initialize interactive selector."""
        self.selector = ObjectSelector(image)
        self.window_name = window_name
        self.selected: Optional[SelectedObject] = None
        self.selection_method = "smart"
        self.tolerance = 20
        
        # Display state
        self.display_image = image.copy()
        self.show_all_contours = False
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - select object
            self.selected = self.selector.select_at_point(
                x, y, 
                tolerance=self.tolerance,
                method=self.selection_method
            )
            self._update_display()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - clear selection
            self.selected = None
            self._update_display()
    
    def _update_display(self):
        """Update the display image."""
        if self.show_all_contours:
            self.display_image = self.selector.draw_all_contours()
        else:
            self.display_image = self.selector.image.copy()
        
        if self.selected:
            # Highlight selected object
            overlay = self.display_image.copy()
            cv2.drawContours(overlay, [self.selected.contour], -1, (0, 255, 0), -1)
            self.display_image = cv2.addWeighted(overlay, 0.3, self.display_image, 0.7, 0)
            
            # Draw contour outline
            cv2.drawContours(self.display_image, [self.selected.contour], -1, (0, 255, 0), 2)
            
            # Draw info
            x, y, w, h = self.selected.bbox
            cv2.rectangle(self.display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            info = f"Area: {self.selected.area} px"
            cv2.putText(self.display_image, info, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def run(self) -> Optional[SelectedObject]:
        """Run interactive selection. Returns selected object."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\nInteractive Object Selection")
        print("=" * 40)
        print("Left click  - Select object")
        print("Right click - Clear selection")
        print("c           - Toggle contour view")
        print("1-3         - Change selection method")
        print("Enter       - Confirm selection")
        print("Esc         - Cancel")
        print("=" * 40)
        
        self._update_display()
        
        while True:
            cv2.imshow(self.window_name, self.display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # Esc
                self.selected = None
                break
            elif key == 13:  # Enter
                break
            elif key == ord('c'):
                self.show_all_contours = not self.show_all_contours
                self._update_display()
            elif key == ord('1'):
                self.selection_method = "smart"
                print("Method: Smart")
            elif key == ord('2'):
                self.selection_method = "flood"
                print("Method: Flood Fill")
            elif key == ord('3'):
                self.selection_method = "contour"
                print("Method: Contour")
        
        cv2.destroyWindow(self.window_name)
        return self.selected


# Test
if __name__ == "__main__":
    import sys
    
    # Create test image with shapes
    test_img = np.ones((400, 600, 3), dtype=np.uint8) * 240
    
    # Draw various shapes
    cv2.circle(test_img, (100, 100), 50, (0, 0, 255), -1)  # Red circle
    cv2.rectangle(test_img, (200, 50), (300, 150), (0, 255, 0), -1)  # Green rect
    cv2.ellipse(test_img, (450, 100), (60, 40), 0, 0, 360, (255, 0, 0), -1)  # Blue ellipse
    
    pts = np.array([[150, 300], [100, 380], [200, 380]], np.int32)
    cv2.fillPoly(test_img, [pts], (0, 255, 255))  # Yellow triangle
    
    cv2.rectangle(test_img, (350, 250), (550, 350), (255, 0, 255), -1)  # Magenta rect
    
    print("Testing ObjectSelector...")
    
    # Test with interactive selector
    selector = InteractiveObjectSelector(test_img)
    selected = selector.run()
    
    if selected:
        print(f"\nSelected object:")
        print(f"  Bounding box: {selected.bbox}")
        print(f"  Center: {selected.center}")
        print(f"  Area: {selected.area} pixels")
        
        # Show selected object
        cv2.imshow("Selected Object", selected.image)
        cv2.waitKey(0)
    else:
        print("No object selected")
    
    cv2.destroyAllWindows()
