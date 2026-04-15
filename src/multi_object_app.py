"""
Multi-Object Extractor and Manipulator
Automatically detects and extracts all objects from an image.
Allows selecting individual objects for gesture-based manipulation.

Features:
    - Auto-detect all objects in image
    - Display extracted objects as thumbnails
    - Click to select any object
    - Apply gesture-based transformations
    - Export individual objects or combined result
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hand_detector import HandDetector, WebcamCapture
from gesture_recognizer import GestureRecognizer, GestureResult


@dataclass
class ExtractedObject:
    """Represents an extracted object from the image."""
    id: int
    name: str
    image: np.ndarray          # BGR image (cropped)
    image_rgba: np.ndarray     # RGBA with transparency
    mask: np.ndarray           # Binary mask
    contour: np.ndarray        # Contour points
    bbox: Tuple[int, int, int, int]  # x, y, w, h in original image
    center: Tuple[int, int]
    area: int
    # Transformation state
    scale: float = 1.0
    rotation: float = 0.0
    translation: Tuple[int, int] = (0, 0)
    flipped: bool = False


class ObjectExtractor:
    """
    Extracts individual objects from an image using various techniques.
    Improved version with better boundary detection.
    """
    
    def __init__(self, image: np.ndarray):
        """Initialize with source image."""
        self.original = image.copy()
        self.height, self.width = image.shape[:2]
        
        # Preprocess
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Detect background color
        self.bg_color = self._detect_background_color()
    
    def _detect_background_color(self) -> np.ndarray:
        """Detect the dominant background color from image edges."""
        # Sample colors from image borders
        border_pixels = []
        
        # Top and bottom rows
        border_pixels.extend(self.original[0, :].tolist())
        border_pixels.extend(self.original[-1, :].tolist())
        # Left and right columns
        border_pixels.extend(self.original[:, 0].tolist())
        border_pixels.extend(self.original[:, -1].tolist())
        # Also sample corners more heavily
        for i in range(min(20, self.height)):
            for j in range(min(20, self.width)):
                border_pixels.append(self.original[i, j].tolist())
                border_pixels.append(self.original[-(i+1), -(j+1)].tolist())
        
        border_pixels = np.array(border_pixels)
        
        # Find most common color using k-means with k=1
        pixels = border_pixels.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        
        return centers[0].astype(np.uint8)
    
    def extract_all_objects(
        self, 
        min_area: int = 1000,
        max_objects: int = 50,
        method: str = "auto"
    ) -> List[ExtractedObject]:
        """
        Extract all objects from the image.
        
        Args:
            min_area: Minimum object area in pixels
            max_objects: Maximum number of objects to extract
            method: Detection method - "auto", "contour", "color", "edge", "background"
        
        Returns:
            List of ExtractedObject
        """
        if method == "auto":
            # Use background subtraction - best for images with clear background
            objects = self._extract_by_background_subtraction(min_area)
            if len(objects) < 2:
                # Fall back to edge-based detection
                objects = self._extract_by_precise_edges(min_area)
        elif method == "contour":
            objects = self._extract_by_precise_edges(min_area)
        elif method == "color":
            objects = self._extract_by_color_regions(min_area)
        elif method == "edge":
            objects = self._extract_by_precise_edges(min_area)
        else:
            objects = self._extract_by_background_subtraction(min_area)
        
        # Limit number of objects
        objects = objects[:max_objects]
        
        # Assign IDs and names
        for i, obj in enumerate(objects):
            obj.id = i
            obj.name = f"Object {i + 1}"
        
        return objects
    
    def _extract_by_background_subtraction(self, min_area: int) -> List[ExtractedObject]:
        """
        Extract objects by identifying and removing background.
        Best for images with uniform background (like the example image).
        """
        objects = []
        
        # Create mask of non-background pixels
        # Calculate color distance from background
        bg_color = self.bg_color.astype(np.float32)
        diff = np.sqrt(np.sum((self.original.astype(np.float32) - bg_color) ** 2, axis=2))
        
        # Threshold to find objects (pixels significantly different from background)
        # Adaptive threshold based on image statistics
        threshold = max(25, np.percentile(diff, 30))
        object_mask = (diff > threshold).astype(np.uint8) * 255
        
        # Also check for edges to include border pixels
        edges = cv2.Canny(self.gray, 30, 100)
        edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Combine: objects are areas different from background OR have strong edges
        combined_mask = cv2.bitwise_or(object_mask, edge_dilated)
        
        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill holes in objects
        combined_mask = self._fill_holes(combined_mask)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_mask, connectivity=8
        )
        
        # Extract each component as an object
        for label_id in range(1, num_labels):  # Skip background (0)
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            
            # Create mask for this component
            component_mask = (labels == label_id).astype(np.uint8) * 255
            
            # Find contour with full boundary
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE  # APPROX_NONE for exact boundary
            )
            
            if contours:
                contour = max(contours, key=cv2.contourArea)
                obj = self._create_object_from_mask(component_mask, contour)
                if obj:
                    objects.append(obj)
        
        # Remove overlapping objects
        objects = self._remove_overlapping(objects, overlap_thresh=0.3)
        objects.sort(key=lambda x: x.area, reverse=True)
        
        return objects
    
    def _extract_by_precise_edges(self, min_area: int) -> List[ExtractedObject]:
        """Extract objects using precise edge detection with flood fill."""
        objects = []
        
        # Multi-scale edge detection
        edges1 = cv2.Canny(self.gray, 30, 80)
        edges2 = cv2.Canny(self.gray, 50, 150)
        edges3 = cv2.Canny(self.gray, 80, 200)
        
        # Combine edges
        edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        
        # Dilate to connect nearby edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Close gaps in edges
        kernel = np.ones((5, 5), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Flood fill from corners to find background
        h, w = edges_closed.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        temp = edges_closed.copy()
        
        # Flood fill from multiple border points
        for x in range(0, w, 10):
            cv2.floodFill(temp, flood_mask, (x, 0), 128)
            cv2.floodFill(temp, flood_mask, (x, h-1), 128)
        for y in range(0, h, 10):
            cv2.floodFill(temp, flood_mask, (0, y), 128)
            cv2.floodFill(temp, flood_mask, (w-1, y), 128)
        
        # Objects are regions NOT filled (not background)
        object_mask = np.where(temp == 128, 0, 255).astype(np.uint8)
        
        # Remove edge pixels
        object_mask = cv2.bitwise_and(object_mask, cv2.bitwise_not(edges_closed))
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Fill holes
        object_mask = self._fill_holes(object_mask)
        
        # Find contours
        contours, _ = cv2.findContours(
            object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Create precise mask from contour
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            obj = self._create_object_from_mask(mask, contour)
            if obj:
                objects.append(obj)
        
        objects.sort(key=lambda x: x.area, reverse=True)
        return objects
    
    def _extract_by_color_regions(self, min_area: int) -> List[ExtractedObject]:
        """Extract objects by segmenting distinct color regions."""
        objects = []
        
        # Use LAB color space for better perceptual color difference
        lab = self.lab.astype(np.float32)
        
        # Reshape for k-means
        pixels = lab.reshape(-1, 3)
        
        # More clusters for better segmentation
        n_clusters = min(20, max(8, self.width * self.height // 30000))
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, labels, centers = cv2.kmeans(
            pixels, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        
        labels = labels.reshape(self.height, self.width)
        
        # Find background labels (from edges)
        edge_labels = np.concatenate([
            labels[0, :], labels[-1, :],
            labels[:, 0], labels[:, -1],
            labels[:5, :].flatten(), labels[-5:, :].flatten()
        ])
        bg_label_counts = np.bincount(edge_labels)
        bg_labels = set(np.where(bg_label_counts > len(edge_labels) * 0.05)[0])
        
        # Process each non-background cluster
        for label_id in range(n_clusters):
            if label_id in bg_labels:
                continue
            
            # Create mask for this color
            color_mask = (labels == label_id).astype(np.uint8) * 255
            
            # Clean up
            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(
                color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                
                mask = np.zeros((self.height, self.width), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                obj = self._create_object_from_mask(mask, contour)
                if obj:
                    objects.append(obj)
        
        # Merge nearby same-object regions and remove duplicates
        objects = self._merge_nearby_objects(objects)
        objects = self._remove_overlapping(objects, overlap_thresh=0.4)
        objects.sort(key=lambda x: x.area, reverse=True)
        
        return objects
    
    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill holes in binary mask."""
        # Flood fill from border
        h, w = mask.shape
        flood = mask.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Fill from all border pixels
        # Top and bottom rows
        for x in range(w):
            if flood[0, x] == 0:
                cv2.floodFill(flood, flood_mask, (x, 0), 128)
            if flood[h-1, x] == 0:
                cv2.floodFill(flood, flood_mask, (x, h-1), 128)
        # Left and right columns
        for y in range(h):
            if flood[y, 0] == 0:
                cv2.floodFill(flood, flood_mask, (0, y), 128)
            if flood[y, w-1] == 0:
                cv2.floodFill(flood, flood_mask, (w-1, y), 128)
        
        # Holes are regions that weren't filled (still 0)
        filled = mask.copy()
        filled[flood == 0] = 255
        
        return filled
    
    def _merge_nearby_objects(self, objects: List[ExtractedObject]) -> List[ExtractedObject]:
        """Merge objects that are very close together (likely same object)."""
        if len(objects) < 2:
            return objects
        
        merged = []
        used = set()
        
        for i, obj1 in enumerate(objects):
            if i in used:
                continue
            
            # Find nearby objects
            to_merge = [obj1]
            x1, y1, w1, h1 = obj1.bbox
            center1 = obj1.center
            
            for j, obj2 in enumerate(objects[i+1:], i+1):
                if j in used:
                    continue
                
                x2, y2, w2, h2 = obj2.bbox
                center2 = obj2.center
                
                # Check if centers are close
                dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                avg_size = (max(w1, h1) + max(w2, h2)) / 2
                
                if dist < avg_size * 0.5:  # Very close
                    to_merge.append(obj2)
                    used.add(j)
            
            if len(to_merge) == 1:
                merged.append(obj1)
            else:
                # Merge masks
                combined_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                for obj in to_merge:
                    combined_mask = cv2.bitwise_or(combined_mask, obj.mask)
                
                # Fill gaps
                kernel = np.ones((5, 5), np.uint8)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                
                # Create new contour
                contours, _ = cv2.findContours(
                    combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    new_obj = self._create_object_from_mask(combined_mask, contour)
                    if new_obj:
                        merged.append(new_obj)
            
            used.add(i)
        
        return merged
    
    def _create_object_from_mask(
        self, 
        mask: np.ndarray, 
        contour: np.ndarray
    ) -> Optional[ExtractedObject]:
        """Create ExtractedObject from a binary mask and its contour."""
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip very small or very large objects
        if w < 15 or h < 15:
            return None
        if w > self.width * 0.95 and h > self.height * 0.95:
            return None
        
        # Ensure mask covers the contour properly
        full_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.drawContours(full_mask, [contour], -1, 255, -1)
        # Include the contour border itself
        cv2.drawContours(full_mask, [contour], -1, 255, 2)
        
        # Calculate center
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        # Extract cropped image
        cropped = self.original[y:y+h, x:x+w].copy()
        cropped_mask = mask[y:y+h, x:x+w]
        
        # Create RGBA image with transparency
        rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = cropped_mask
        
        return ExtractedObject(
            id=0,
            name="",
            image=cropped,
            image_rgba=rgba,
            mask=mask,
            contour=contour,
            bbox=(x, y, w, h),
            center=(cx, cy),
            area=cv2.contourArea(contour)
        )
    
    def _remove_overlapping(
        self, 
        objects: List[ExtractedObject], 
        overlap_thresh: float = 0.5
    ) -> List[ExtractedObject]:
        """Remove overlapping objects, keeping larger ones."""
        if not objects:
            return objects
        
        # Sort by area descending
        objects.sort(key=lambda x: x.area, reverse=True)
        
        keep = []
        for obj in objects:
            is_overlapping = False
            for kept in keep:
                # Check IoU
                iou = self._calculate_iou(obj.bbox, kept.bbox)
                if iou > overlap_thresh:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                keep.append(obj)
        
        return keep
    
    def _calculate_iou(self, box1, box2) -> float:
        """Calculate Intersection over Union of two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class MultiObjectManipulator:
    """
    UI Application for extracting and manipulating multiple objects.
    Supports both gesture-based and manual object selection.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.root = tk.Tk()
        self.root.title("Multi-Object Extractor & Manipulator")
        
        # Get screen dimensions and set window size to fit
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Use 90% of screen or max reasonable size
        win_width = min(1500, int(screen_width * 0.9))
        win_height = min(950, int(screen_height * 0.85))
        
        # Center the window
        x = (screen_width - win_width) // 2
        y = (screen_height - win_height) // 2 - 30
        
        self.root.geometry(f"{win_width}x{win_height}+{x}+{y}")
        self.root.minsize(1000, 600)
        self.root.configure(bg='#1e1e1e')
        
        # State
        self.original_image = None
        self.extractor = None
        self.objects: List[ExtractedObject] = []
        self.selected_object: ExtractedObject = None
        self.selected_index = -1
        self.object_thumbnails = {}
        
        # Gesture control
        self.gesture_active = False
        self.detector = None
        self.recognizer = None
        self.webcam = None
        self.current_gesture = GestureResult()
        
        # Gesture selection state
        self.gesture_mode = "transform"  # "select" or "transform"
        self.pointing_position = None  # (x, y) normalized pointing position
        self.hover_object = None  # Object being hovered by gesture
        self.selection_cooldown = 0  # Frames to wait before allowing new selection
        self.swipe_start_x = None  # For swipe gesture detection
        self.last_index_tip_x = None
        self.pinch_select_active = False
        self.thumb_index_dist_history = []
        self.last_palm_facing = None  # Track palm orientation for flip detection
        self.was_fist = False  # Track fist state for selection
        
        # Transform mode: "object" or "image"
        self.transform_target = tk.StringVar(value="object")
        self.full_image_scale = 1.0
        self.full_image_rotation = 0.0
        self.full_image_flipped = False
        
        # Settings
        self.min_area = tk.IntVar(value=1000)
        self.extraction_method = tk.StringVar(value="auto")
        
        # Build UI
        self._build_ui()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _build_ui(self):
        """Build the user interface."""
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabel', background='#1e1e1e', foreground='white')
        style.configure('TButton', padding=6)
        style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('Subheader.TLabel', font=('Helvetica', 11, 'bold'))
        
        # Main container
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top bar
        self._build_top_bar(main)
        
        # Content area
        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left: Objects panel
        objects_panel = ttk.Frame(content, width=180)
        objects_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        objects_panel.pack_propagate(False)
        self._build_objects_panel(objects_panel)
        
        # Center: Main image view
        image_panel = ttk.Frame(content)
        image_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_image_panel(image_panel)
        
        # Right: Controls and webcam (with scrollbar)
        control_panel = ttk.Frame(content, width=300)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        control_panel.pack_propagate(False)
        
        # Add scrollable canvas for control panel
        control_canvas = tk.Canvas(control_panel, bg='#1e1e1e', highlightthickness=0, width=280)
        control_scrollbar = ttk.Scrollbar(control_panel, orient="vertical", 
                                          command=control_canvas.yview)
        self.control_inner = ttk.Frame(control_canvas, width=270)
        
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        control_canvas.create_window((0, 0), window=self.control_inner, anchor=tk.NW)
        self.control_inner.bind("<Configure>", 
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        self._build_control_panel(self.control_inner)
        
        # Status bar
        self.status_var = tk.StringVar(value="Upload an image to get started")
        ttk.Label(main, textvariable=self.status_var).pack(fill=tk.X, pady=(10, 0))
    
    def _build_top_bar(self, parent):
        """Build top toolbar."""
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        ttk.Label(toolbar, text="🎯 Multi-Object Extractor & Manipulator", 
                 style='Header.TLabel').pack(side=tk.LEFT)
        
        # Buttons
        btn_frame = ttk.Frame(toolbar)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="📁 Upload Image", 
                  command=self._upload_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="🔍 Extract Objects", 
                  command=self._extract_objects).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="💾 Save Result", 
                  command=self._save_result).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="🔄 Reset", 
                  command=self._reset_all).pack(side=tk.LEFT, padx=3)
    
    def _build_objects_panel(self, parent):
        """Build objects list panel."""
        ttk.Label(parent, text="EXTRACTED OBJECTS", 
                 style='Subheader.TLabel').pack(pady=(0, 10))
        
        # Object count
        self.obj_count_var = tk.StringVar(value="0 objects")
        ttk.Label(parent, textvariable=self.obj_count_var).pack()
        
        # Scrollable object list
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.objects_canvas = tk.Canvas(canvas_frame, bg='#2b2b2b', 
                                        highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, 
                                 command=self.objects_canvas.yview)
        
        self.objects_canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.objects_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas for object thumbnails
        self.objects_frame = ttk.Frame(self.objects_canvas)
        self.objects_canvas.create_window((0, 0), window=self.objects_frame, 
                                          anchor=tk.NW)
        
        self.objects_frame.bind("<Configure>", 
            lambda e: self.objects_canvas.configure(
                scrollregion=self.objects_canvas.bbox("all")))
        
        # Export selected button
        ttk.Button(parent, text="📤 Export Selected", 
                  command=self._export_selected).pack(fill=tk.X, pady=5)
        ttk.Button(parent, text="📤 Export All", 
                  command=self._export_all).pack(fill=tk.X)
    
    def _build_image_panel(self, parent):
        """Build main image display."""
        ttk.Label(parent, text="IMAGE VIEW", style='Subheader.TLabel').pack()
        
        # Canvas for image
        self.canvas = tk.Canvas(parent, bg='#2b2b2b', highlightthickness=1,
                               highlightbackground='#444')
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Bind mouse
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", lambda e: self._update_display())
    
    def _build_control_panel(self, parent):
        """Build control panel with settings and webcam."""
        # Extraction settings
        settings_frame = ttk.LabelFrame(parent, text="Extraction Settings")
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="Method:").pack(anchor=tk.W, padx=5)
        methods = [("Auto", "auto"), ("Contour", "contour"), 
                  ("Color", "color"), ("Edge", "edge")]
        for text, val in methods:
            ttk.Radiobutton(settings_frame, text=text, value=val,
                           variable=self.extraction_method).pack(anchor=tk.W, padx=20)
        
        ttk.Label(settings_frame, text="Min Object Size:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        ttk.Scale(settings_frame, from_=100, to=5000, variable=self.min_area,
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5)
        
        # Selected object info
        self.obj_info_frame = ttk.LabelFrame(parent, text="Selected Object")
        self.obj_info_frame.pack(fill=tk.X, pady=10)
        
        self.obj_info_label = ttk.Label(self.obj_info_frame, text="None selected")
        self.obj_info_label.pack(padx=5, pady=5)
        
        # Transform controls
        trans_frame = ttk.LabelFrame(parent, text="Transform Controls")
        trans_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(trans_frame, text="Scale:").pack(anchor=tk.W, padx=5)
        self.scale_var = tk.DoubleVar(value=1.0)
        ttk.Scale(trans_frame, from_=0.2, to=3.0, variable=self.scale_var,
                 command=self._on_transform_change).pack(fill=tk.X, padx=5)
        
        ttk.Label(trans_frame, text="Rotation:").pack(anchor=tk.W, padx=5)
        self.rotation_var = tk.DoubleVar(value=0.0)
        ttk.Scale(trans_frame, from_=-180, to=180, variable=self.rotation_var,
                 command=self._on_transform_change).pack(fill=tk.X, padx=5)
        
        self.flip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(trans_frame, text="Flip Horizontal", variable=self.flip_var,
                       command=self._on_transform_change).pack(anchor=tk.W, padx=5, pady=5)
        
        # Transform Mode toggle
        mode_frame = ttk.LabelFrame(parent, text="Transform Mode")
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(mode_frame, text="🎯 Individual Object", value="object",
                       variable=self.transform_target,
                       command=self._on_mode_change).pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(mode_frame, text="🖼 Full Image", value="image",
                       variable=self.transform_target,
                       command=self._on_mode_change).pack(anchor=tk.W, padx=10, pady=2)
        
        self.mode_info_label = ttk.Label(mode_frame, text="Transforms apply to selected object",
                                         font=('Helvetica', 8), foreground='gray')
        self.mode_info_label.pack(padx=10, pady=2)
        
        # Gesture control
        gesture_frame = ttk.LabelFrame(parent, text="Gesture Control")
        gesture_frame.pack(fill=tk.X, pady=10)
        
        # Webcam preview
        self.webcam_label = ttk.Label(gesture_frame)
        self.webcam_label.pack(pady=5)
        self._show_webcam_placeholder()
        
        self.gesture_btn_text = tk.StringVar(value="▶ Start Gesture Control")
        ttk.Button(gesture_frame, textvariable=self.gesture_btn_text,
                  command=self._toggle_gesture).pack(fill=tk.X, padx=5, pady=5)
        
        # Gesture mode selection
        mode_frame = ttk.Frame(gesture_frame)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        self.gesture_mode_var = tk.StringVar(value="select")
        ttk.Radiobutton(mode_frame, text="Select", value="select",
                       variable=self.gesture_mode_var,
                       command=self._on_gesture_mode_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Transform", value="transform",
                       variable=self.gesture_mode_var,
                       command=self._on_gesture_mode_change).pack(side=tk.LEFT, padx=5)
        
        # Gesture instructions
        inst_frame = ttk.LabelFrame(gesture_frame, text="🎮 GESTURE CONTROLS")
        inst_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.gesture_instructions = ttk.Label(inst_frame, text=
            "UNIVERSAL:\n"
            "✌️ Peace → Switch Object/Image\n"
            "🤟 Rock (Index+Pinky) → Reset\n\n"
            "SELECT MODE:\n"
            "☝️ Point → Move cursor\n"
            "✊ Fist → Select/Transform\n\n"
            "TRANSFORM MODE:\n"
            "🤏 Pinch → Scale\n"
            "🖐️ Tilt → Rotate\n"
            "🤙 Pinky+Thumb → Flip",
            wraplength=300, justify=tk.LEFT)
        self.gesture_instructions.pack(padx=5, pady=5)
        
        # Gesture info
        self.gesture_labels = {}
        for name in ["Mode", "Hover", "Selected", "Action"]:
            row = ttk.Frame(gesture_frame)
            row.pack(fill=tk.X, padx=5)
            ttk.Label(row, text=f"{name}:", width=10).pack(side=tk.LEFT)
            lbl = ttk.Label(row, text="-")
            lbl.pack(side=tk.LEFT)
            self.gesture_labels[name] = lbl
    
    def _show_webcam_placeholder(self):
        """Show placeholder for webcam."""
        img = np.zeros((180, 240, 3), dtype=np.uint8)
        cv2.putText(img, "Webcam Off", (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (80, 80, 80), 2)
        self._update_webcam_frame(img)
    
    def _on_gesture_mode_change(self):
        """Handle gesture mode change."""
        mode = self.gesture_mode_var.get()
        self.gesture_mode = mode
        
        if mode == "select":
            self.gesture_instructions.config(text=
                "UNIVERSAL:\n"
                "✌️ Peace → Switch Object/Image\n"
                "🤟 Rock → Reset transforms\n\n"
                "SELECT MODE:\n"
                "☝️ Point → Move cursor\n"
                "✊ Fist → Select/Transform")
        else:
            self.gesture_instructions.config(text=
                "UNIVERSAL:\n"
                "✌️ Peace → Switch Object/Image\n"
                "🤟 Rock → Reset transforms\n\n"
                "TRANSFORM MODE:\n"
                "🤏 Pinch → Scale\n"
                "🖐️ Tilt → Rotate\n"
                "🤙 Pinky+Thumb → Flip")
        
        self.status_var.set(f"Gesture mode: {mode.upper()}")
    
    def _update_webcam_frame(self, frame):
        """Update webcam preview."""
        frame = cv2.resize(frame, (200, 150))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.webcam_photo = ImageTk.PhotoImage(img)
        self.webcam_label.configure(image=self.webcam_photo)
    
    def _upload_image(self):
        """Upload image file."""
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                    ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        
        if path:
            image = cv2.imread(path)
            if image is not None:
                # Resize if too large
                max_dim = 1200
                h, w = image.shape[:2]
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    image = cv2.resize(image, None, fx=scale, fy=scale)
                
                self.original_image = image
                self.extractor = ObjectExtractor(image)
                self.objects = []
                self.selected_object = None
                self._update_objects_list()
                self._update_display()
                self.status_var.set(f"Loaded: {os.path.basename(path)} - Click 'Extract Objects'")
            else:
                messagebox.showerror("Error", "Could not load image")
    
    def _extract_objects(self):
        """Extract all objects from image."""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please upload an image first")
            return
        
        self.status_var.set("Extracting objects...")
        self.root.update()
        
        try:
            self.objects = self.extractor.extract_all_objects(
                min_area=self.min_area.get(),
                method=self.extraction_method.get()
            )
            
            self._update_objects_list()
            self._update_display()
            
            self.status_var.set(f"Found {len(self.objects)} objects - Click to select")
            self.obj_count_var.set(f"{len(self.objects)} objects")
            
        except Exception as e:
            messagebox.showerror("Error", f"Extraction failed: {e}")
            self.status_var.set("Extraction failed")
    
    def _update_objects_list(self):
        """Update the objects thumbnail list."""
        # Clear existing
        for widget in self.objects_frame.winfo_children():
            widget.destroy()
        self.object_thumbnails.clear()
        
        # Add thumbnails
        for obj in self.objects:
            frame = tk.Frame(self.objects_frame, bg='#3b3b3b', cursor='hand2')
            frame.pack(fill=tk.X, pady=3, padx=5)
            
            # Thumbnail
            thumb = self._create_thumbnail(obj.image, size=60)
            thumb_label = tk.Label(frame, image=thumb, bg='#3b3b3b')
            thumb_label.image = thumb
            thumb_label.pack(side=tk.LEFT, padx=5, pady=5)
            
            # Info
            info_frame = tk.Frame(frame, bg='#3b3b3b')
            info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            tk.Label(info_frame, text=obj.name, bg='#3b3b3b', fg='white',
                    font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
            tk.Label(info_frame, text=f"{obj.bbox[2]}x{obj.bbox[3]} px", 
                    bg='#3b3b3b', fg='#aaa').pack(anchor=tk.W)
            
            # Bind click
            for widget in [frame, thumb_label, info_frame]:
                widget.bind("<Button-1>", lambda e, o=obj: self._select_object(o))
            
            self.object_thumbnails[obj.id] = frame
    
    def _create_thumbnail(self, image: np.ndarray, size: int = 60) -> ImageTk.PhotoImage:
        """Create thumbnail from OpenCV image."""
        h, w = image.shape[:2]
        scale = size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create square background
        thumb = np.ones((size, size, 3), dtype=np.uint8) * 50
        x_off = (size - new_w) // 2
        y_off = (size - new_h) // 2
        thumb[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))
    
    def _select_object(self, obj: ExtractedObject):
        """Select an object."""
        self.selected_object = obj
        
        # Update highlight in list
        for oid, frame in self.object_thumbnails.items():
            if oid == obj.id:
                frame.configure(bg='#4a90d9')
                for child in frame.winfo_children():
                    try:
                        child.configure(bg='#4a90d9')
                    except:
                        pass
            else:
                frame.configure(bg='#3b3b3b')
                for child in frame.winfo_children():
                    try:
                        child.configure(bg='#3b3b3b')
                    except:
                        pass
        
        # Update info
        x, y, w, h = obj.bbox
        self.obj_info_label.config(text=f"{obj.name}\nSize: {w}x{h}\nArea: {obj.area} px")
        
        # Reset transforms
        self.scale_var.set(obj.scale)
        self.rotation_var.set(obj.rotation)
        self.flip_var.set(obj.flipped)
        
        self._update_display()
        self.status_var.set(f"Selected: {obj.name}")
    
    def _deselect_object(self):
        """Deselect the current object."""
        if self.selected_object is None:
            return
        
        # Reset highlight in list
        for oid, frame in self.object_thumbnails.items():
            frame.configure(bg='#3b3b3b')
            for child in frame.winfo_children():
                try:
                    child.configure(bg='#3b3b3b')
                except:
                    pass
        
        self.selected_object = None
        self.selected_index = -1
        self.hover_object = None
        self.pointing_position = None
        
        # Clear info
        self.obj_info_label.config(text="No object selected")
        
        # Reset transforms
        self.scale_var.set(1.0)
        self.rotation_var.set(0)
        self.flip_var.set(False)
        
        self._update_display()
        self.status_var.set("Object released")
    
    def _on_canvas_click(self, event):
        """Handle click on canvas to select object."""
        if not self.objects or self.original_image is None:
            return
        
        # Convert to image coordinates
        x, y = self._canvas_to_image(event.x, event.y)
        if x is None:
            return
        
        # Find object at this point
        for obj in self.objects:
            if cv2.pointPolygonTest(obj.contour, (x, y), False) >= 0:
                self._select_object(obj)
                return
        
        # Click outside objects - deselect
        self.selected_object = None
        self._update_display()
    
    def _canvas_to_image(self, cx, cy):
        """Convert canvas coords to image coords."""
        if self.original_image is None:
            return None, None
        
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        ih, iw = self.original_image.shape[:2]
        
        scale = min(cw / iw, ch / ih) * 0.95
        sw, sh = int(iw * scale), int(ih * scale)
        
        ox = (cw - sw) // 2
        oy = (ch - sh) // 2
        
        ix = int((cx - ox) / scale)
        iy = int((cy - oy) / scale)
        
        if 0 <= ix < iw and 0 <= iy < ih:
            return ix, iy
        return None, None
    
    def _on_transform_change(self, *args):
        """Handle transform slider changes."""
        mode = self.transform_target.get()
        scale = self.scale_var.get()
        rotation = self.rotation_var.get()
        flip = self.flip_var.get()
        
        print(f"Transform change: mode={mode}, scale={scale:.2f}, rot={rotation:.1f}, flip={flip}")
        
        if mode == "object":
            # Apply to selected object
            if self.selected_object:
                self.selected_object.scale = scale
                self.selected_object.rotation = rotation
                self.selected_object.flipped = flip
        else:
            # Apply to full image
            self.full_image_scale = scale
            self.full_image_rotation = rotation
            self.full_image_flipped = flip
        
        self._update_display()
    
    def _on_mode_change(self):
        """Handle transform mode change."""
        mode = self.transform_target.get()
        print(f"Mode changed to: {mode}")
        
        if mode == "object":
            self.mode_info_label.config(text="Transforms apply to selected object")
            # Restore object transforms to sliders
            if self.selected_object:
                self.scale_var.set(self.selected_object.scale)
                self.rotation_var.set(self.selected_object.rotation)
                self.flip_var.set(self.selected_object.flipped)
            else:
                self.scale_var.set(1.0)
                self.rotation_var.set(0.0)
                self.flip_var.set(False)
        else:
            self.mode_info_label.config(text="Transforms apply to entire image")
            # Restore full image transforms to sliders
            self.scale_var.set(self.full_image_scale)
            self.rotation_var.set(self.full_image_rotation)
            self.flip_var.set(self.full_image_flipped)
        
        self._update_display()
    
    def _update_display(self):
        """Update main image display."""
        if self.original_image is None:
            return
        
        mode = self.transform_target.get()
        print(f"Update display: mode={mode}")
        
        if mode == "image":
            # Full image transform mode
            print(f"Applying full image: scale={self.full_image_scale}, rot={self.full_image_rotation}, flip={self.full_image_flipped}")
            display = self._apply_full_image_transform()
        else:
            # Object transform mode
            display = self.original_image.copy()
            
            # Draw all objects with highlights
            for obj in self.objects:
                color = (0, 255, 0) if obj == self.selected_object else (100, 100, 100)
                thickness = 3 if obj == self.selected_object else 1
                cv2.drawContours(display, [obj.contour], -1, color, thickness)
            
            # Apply transformation to selected object
            if self.selected_object:
                display = self._apply_transform(display, self.selected_object)
                
                # Draw bounding box
                x, y, w, h = self.selected_object.bbox
                cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display on canvas
        self._show_on_canvas(display)
    
    def _apply_full_image_transform(self) -> np.ndarray:
        """Apply transformation to the full image."""
        img = self.original_image.copy()
        h, w = img.shape[:2]
        
        # Scale
        if self.full_image_scale != 1.0:
            new_w = int(w * self.full_image_scale)
            new_h = int(h * self.full_image_scale)
            img = cv2.resize(img, (new_w, new_h))
            
            # Center on original size canvas
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            # Fill with average color
            canvas[:] = cv2.mean(self.original_image)[:3]
            
            # Calculate offset to center
            offset_x = (w - new_w) // 2
            offset_y = (h - new_h) // 2
            
            # Handle cases where scaled image is larger than canvas
            src_x = max(0, -offset_x)
            src_y = max(0, -offset_y)
            dst_x = max(0, offset_x)
            dst_y = max(0, offset_y)
            
            copy_w = min(new_w - src_x, w - dst_x)
            copy_h = min(new_h - src_y, h - dst_y)
            
            if copy_w > 0 and copy_h > 0:
                canvas[dst_y:dst_y+copy_h, dst_x:dst_x+copy_w] = img[src_y:src_y+copy_h, src_x:src_x+copy_w]
            
            img = canvas
        
        # Rotation
        if self.full_image_rotation != 0:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, self.full_image_rotation, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Flip
        if self.full_image_flipped:
            img = cv2.flip(img, 1)
        
        return img
    
    def _apply_transform(self, image: np.ndarray, obj: ExtractedObject) -> np.ndarray:
        """Apply transformation to object in image."""
        result = image.copy()
        
        # Get object region
        x, y, w, h = obj.bbox
        obj_img = obj.image.copy()
        
        # Apply transformations
        # Scale
        if obj.scale != 1.0:
            new_w = int(w * obj.scale)
            new_h = int(h * obj.scale)
            obj_img = cv2.resize(obj_img, (new_w, new_h))
        
        # Rotation
        if obj.rotation != 0:
            center = (obj_img.shape[1] // 2, obj_img.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, obj.rotation, 1.0)
            obj_img = cv2.warpAffine(obj_img, M, (obj_img.shape[1], obj_img.shape[0]),
                                     borderMode=cv2.BORDER_REPLICATE)
        
        # Flip
        if obj.flipped:
            obj_img = cv2.flip(obj_img, 1)
        
        # Place back (centered on original position)
        oh, ow = obj_img.shape[:2]
        cx, cy = obj.center
        
        # Calculate new position
        new_x = cx - ow // 2
        new_y = cy - oh // 2
        
        # Mask for placement
        obj_mask = cv2.resize(obj.mask[y:y+h, x:x+w], (ow, oh))
        if obj.rotation != 0:
            obj_mask = cv2.warpAffine(obj_mask, M, (ow, oh))
        if obj.flipped:
            obj_mask = cv2.flip(obj_mask, 1)
        
        # Erase original object area
        cv2.drawContours(result, [obj.contour], -1, 
                        tuple(int(c) for c in cv2.mean(image)[:3]), -1)
        
        # Place transformed object
        for yy in range(oh):
            for xx in range(ow):
                py, px = new_y + yy, new_x + xx
                if 0 <= py < result.shape[0] and 0 <= px < result.shape[1]:
                    if obj_mask[yy, xx] > 127:
                        result[py, px] = obj_img[yy, xx]
        
        return result
    
    def _show_on_canvas(self, image: np.ndarray):
        """Display image on canvas."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        if cw < 10 or ch < 10:
            return
        
        ih, iw = image.shape[:2]
        scale = min(cw / iw, ch / ih) * 0.95
        
        resized = cv2.resize(image, (int(iw * scale), int(ih * scale)))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        self.canvas_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.canvas_photo)
    
    def _toggle_gesture(self):
        """Toggle gesture control."""
        if self.gesture_active:
            self._stop_gesture()
        else:
            self._start_gesture()
    
    def _start_gesture(self):
        """Start gesture control."""
        try:
            self.detector = HandDetector(max_hands=1)
            self.recognizer = GestureRecognizer()
            self.webcam = WebcamCapture()
            
            self.gesture_active = True
            self.gesture_btn_text.set("⏹ Stop Gesture")
            
            threading.Thread(target=self._gesture_loop, daemon=True).start()
            self.status_var.set("Gesture control active!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start: {e}")
    
    def _stop_gesture(self):
        """Stop gesture control."""
        self.gesture_active = False
        self.gesture_btn_text.set("▶ Start Gesture Control")
        
        if self.webcam:
            self.webcam.release()
        if self.detector:
            self.detector.release()
        
        self._show_webcam_placeholder()
    
    def _gesture_loop(self):
        """Gesture processing loop - ALL operations via gestures."""
        import math
        import time
        
        last_hover_obj = None
        frame_count = 0
        
        while self.gesture_active:
            try:
                ret, frame = self.webcam.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                landmarks = self.detector.detect(frame)
                
                action_text = "-"
                frame_count += 1
                
                if landmarks:
                    # Get key landmark positions
                    index_tip = landmarks[8]
                    thumb_tip = landmarks[4]
                    index_mcp = landmarks[5]
                    wrist = landmarks[0]
                    middle_tip = landmarks[12]
                    ring_tip = landmarks[16]
                    pinky_tip = landmarks[20]
                    
                    # Calculate pinch distance
                    pinch_dist = math.sqrt(
                        (thumb_tip.x - index_tip.x)**2 + 
                        (thumb_tip.y - index_tip.y)**2
                    )
                    
                    # Detect finger states
                    index_extended = index_tip.y < index_mcp.y
                    middle_extended = middle_tip.y < landmarks[9].y
                    ring_extended = ring_tip.y < landmarks[13].y
                    pinky_extended = pinky_tip.y < landmarks[17].y
                    
                    middle_curled = not middle_extended
                    ring_curled = not ring_extended
                    pinky_curled = not pinky_extended
                    
                    # Gesture detection
                    is_pointing = index_extended and middle_curled and ring_curled and pinky_curled
                    is_pinching = pinch_dist < 0.06
                    
                    # Peace sign (index + middle) -> Switch transform target
                    is_peace = index_extended and middle_extended and ring_curled and pinky_curled
                    
                    # Three fingers -> Switch to Transform mode
                    is_three_fingers = index_extended and middle_extended and ring_extended and pinky_curled
                    
                    # Open palm (all extended)
                    all_extended = index_extended and middle_extended and ring_extended and pinky_extended
                    
                    # Fist (all curled)
                    is_fist = not index_extended and not middle_extended and not ring_extended and not pinky_extended
                    
                    # Rock sign (index + pinky) -> Reset
                    is_rock = index_extended and pinky_extended and middle_curled and ring_curled
                    
                    # Pinky-thumb touch (flip)
                    pinky_thumb_dist = math.sqrt(
                        (thumb_tip.x - pinky_tip.x)**2 + 
                        (thumb_tip.y - pinky_tip.y)**2
                    )
                    is_pinky_thumb_touch = pinky_thumb_dist < 0.08
                    
                    # Palm orientation
                    middle_mcp = landmarks[9]
                    palm_facing = "front" if middle_mcp.x > wrist.x else "back"
                    palm_flipped = False
                    if self.last_palm_facing is not None and self.last_palm_facing != palm_facing:
                        palm_flipped = True
                    self.last_palm_facing = palm_facing
                    
                    # Cooldown
                    if self.selection_cooldown > 0:
                        self.selection_cooldown -= 1
                    
                    gesture_mode = self.gesture_mode_var.get()
                    transform_target = self.transform_target.get()
                    
                    # === UNIVERSAL GESTURES ===
                    
                    # PEACE: Switch Object <-> Image
                    if is_peace and not is_pinching and self.selection_cooldown == 0:
                        new_target = "image" if transform_target == "object" else "object"
                        self.transform_target.set(new_target)
                        self.root.after(0, self._on_mode_change)
                        self.selection_cooldown = 40
                        action_text = f"TARGET → {new_target.upper()}"
                    
                    # ROCK: Reset transforms
                    elif is_rock and self.selection_cooldown == 0:
                        self.root.after(0, self._reset_transforms_only)
                        self.selection_cooldown = 40
                        action_text = "RESET!"
                    
                    # THREE FINGERS: Go to Transform mode
                    elif is_three_fingers and gesture_mode == "select" and self.selection_cooldown == 0:
                        self.gesture_mode_var.set("transform")
                        self.root.after(0, self._on_gesture_mode_change)
                        self.selection_cooldown = 30
                        action_text = "→ TRANSFORM"
                    
                    elif gesture_mode == "select":
                        # === SELECT MODE ===
                        
                        if is_pointing and self.objects and transform_target == "object":
                            px = int(index_tip.x * (self.original_image.shape[1] if self.original_image is not None else w))
                            py = int(index_tip.y * (self.original_image.shape[0] if self.original_image is not None else h))
                            self.pointing_position = (px, py)
                            
                            hover_obj = None
                            for obj in self.objects:
                                if cv2.pointPolygonTest(obj.contour, (px, py), False) >= 0:
                                    hover_obj = obj
                                    break
                            self.hover_object = hover_obj
                            action_text = f"Point ({px},{py})"
                            cv2.circle(frame, (int(index_tip.x * w), int(index_tip.y * h)), 10, (0, 255, 0), -1)
                        
                        # FIST: Select or go to transform
                        if is_fist and not self.was_fist and self.selection_cooldown == 0:
                            if self.hover_object:
                                self.root.after(0, lambda o=self.hover_object: self._select_object(o))
                                action_text = f"SELECT: {self.hover_object.name}"
                            self.gesture_mode_var.set("transform")
                            self.root.after(0, self._on_gesture_mode_change)
                            self.selection_cooldown = 30
                        
                        self.was_fist = is_fist
                        
                        # Swipe to cycle
                        if is_pointing and self.objects and transform_target == "object":
                            current_x = index_tip.x
                            if self.last_index_tip_x is not None:
                                dx = current_x - self.last_index_tip_x
                                if abs(dx) > 0.15 and self.selection_cooldown == 0:
                                    self._cycle_selection(1 if dx > 0 else -1)
                                    action_text = "Swipe →" if dx > 0 else "Swipe ←"
                                    self.selection_cooldown = 20
                            self.last_index_tip_x = current_x
                        else:
                            self.last_index_tip_x = None
                        
                        hover_changed = (self.hover_object != last_hover_obj)
                        last_hover_obj = self.hover_object
                        if hover_changed or frame_count % 5 == 0:
                            self.root.after(0, self._update_gesture_select_info, self.hover_object, action_text)
                    
                    else:
                        # === TRANSFORM MODE ===
                        gesture = self.recognizer.recognize(landmarks, self.webcam.width, self.webcam.height)
                        
                        if transform_target == "image":
                            # Full image transform
                            self.full_image_scale = gesture.scale_factor
                            self.full_image_rotation = gesture.rotation_angle
                            action_text = f"🖼 S:{gesture.scale_factor:.2f} R:{gesture.rotation_angle:.0f}°"
                            self.root.after(0, self._sync_from_gesture_image, gesture, action_text)
                            
                            if is_pinky_thumb_touch and self.selection_cooldown == 0:
                                self.full_image_flipped = not self.full_image_flipped
                                self.selection_cooldown = 30
                                action_text = "🖼 FLIP!"
                                self.root.after(0, self._update_display)
                        
                        elif self.selected_object:
                            # Object transform
                            self.selected_object.scale = gesture.scale_factor
                            self.selected_object.rotation = gesture.rotation_angle
                            action_text = f"🎯 S:{gesture.scale_factor:.2f} R:{gesture.rotation_angle:.0f}°"
                            self.root.after(0, self._sync_from_gesture, gesture, action_text)
                            
                            if is_pinky_thumb_touch and self.selection_cooldown == 0:
                                self.selected_object.flipped = not self.selected_object.flipped
                                self.selection_cooldown = 30
                                action_text = "🎯 FLIP!"
                                self.root.after(0, self._update_display)
                        
                        # PALM FLIP: Back to select
                        if palm_flipped and self.selection_cooldown == 0:
                            if transform_target == "object" and self.selected_object:
                                self.root.after(0, self._deselect_object)
                            self.gesture_mode_var.set("select")
                            self.root.after(0, self._on_gesture_mode_change)
                            self.selection_cooldown = 30
                            action_text = "→ SELECT"
                        
                        # OPEN PALM: Back to select (keep object)
                        if all_extended and not is_pinching and self.selection_cooldown == 0:
                            self.gesture_mode_var.set("select")
                            self.root.after(0, self._on_gesture_mode_change)
                            self.selection_cooldown = 30
                            action_text = "→ SELECT"
                
                # Draw on frame
                self.detector.draw_landmarks(frame, landmarks)
                target_text = f"Target: {self.transform_target.get().upper()}"
                gesture_text = f"Mode: {self.gesture_mode_var.get().upper()}"
                cv2.putText(frame, target_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, action_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                self.root.after(0, lambda f=frame.copy(): self._update_webcam_frame(f))
                time.sleep(0.033)
                
            except Exception as e:
                print(f"Gesture error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.05)
    
    def _reset_transforms_only(self):
        """Reset transforms without clearing objects."""
        if self.transform_target.get() == "image":
            self.full_image_scale = 1.0
            self.full_image_rotation = 0.0
            self.full_image_flipped = False
        elif self.selected_object:
            self.selected_object.scale = 1.0
            self.selected_object.rotation = 0.0
            self.selected_object.flipped = False
        
        self.scale_var.set(1.0)
        self.rotation_var.set(0.0)
        self.flip_var.set(False)
        self._update_display()
        self.status_var.set("Transforms reset")
    
    def _sync_from_gesture_image(self, gesture, action_text="-"):
        """Sync UI from gesture for full image mode."""
        self.scale_var.set(gesture.scale_factor)
        self.rotation_var.set(gesture.rotation_angle)
        self.flip_var.set(self.full_image_flipped)
        
        self.gesture_labels["Mode"].config(text="🖼 IMAGE", foreground="cyan")
        self.gesture_labels["Hover"].config(text="-", foreground="gray")
        self.gesture_labels["Selected"].config(text="Full Image")
        self.gesture_labels["Action"].config(text=action_text)
        
        self._update_display()
    
    def _cycle_selection(self, direction: int):
        """Cycle through objects in the given direction."""
        if not self.objects:
            return
        
        if self.selected_object is None:
            self.selected_index = 0
        else:
            self.selected_index = (self.selected_index + direction) % len(self.objects)
        
        obj = self.objects[self.selected_index]
        self.root.after(0, lambda: self._select_object(obj))
    
    def _update_gesture_select_info(self, hover_obj, action_text):
        """Update gesture info labels for selection mode."""
        self.gesture_labels["Mode"].config(text="SELECT", foreground="cyan")
        self.gesture_labels["Hover"].config(
            text=hover_obj.name if hover_obj else "-",
            foreground="yellow" if hover_obj else "gray"
        )
        self.gesture_labels["Selected"].config(
            text=self.selected_object.name if self.selected_object else "-"
        )
        self.gesture_labels["Action"].config(text=action_text)
        
        # Update display to show hover highlight
        if self.original_image is not None:
            self._update_display_with_hover(hover_obj)
    
    def _update_display_with_hover(self, hover_obj):
        """Update display with hover highlight and cursor."""
        if self.original_image is None:
            return
        
        display = self.original_image.copy()
        
        # Draw all objects
        for obj in self.objects:
            if obj == self.selected_object:
                color = (0, 255, 0)  # Green for selected
                thickness = 3
            elif obj == hover_obj:
                color = (0, 255, 255)  # Yellow for hover
                thickness = 2
            else:
                color = (100, 100, 100)  # Gray for others
                thickness = 1
            cv2.drawContours(display, [obj.contour], -1, color, thickness)
        
        # Apply transformation to selected object
        if self.selected_object:
            display = self._apply_transform(display, self.selected_object)
        
        # Draw cursor at pointing position
        if self.pointing_position:
            px, py = self.pointing_position
            h, w = display.shape[:2]
            
            # Ensure cursor is within image bounds
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))
            
            # Draw crosshair cursor
            cursor_size = 20
            cursor_color = (0, 255, 255) if hover_obj else (255, 255, 255)  # Yellow if hovering, white otherwise
            
            # Horizontal line
            cv2.line(display, (px - cursor_size, py), (px + cursor_size, py), cursor_color, 2)
            # Vertical line
            cv2.line(display, (px, py - cursor_size), (px, py + cursor_size), cursor_color, 2)
            # Center circle
            cv2.circle(display, (px, py), 5, cursor_color, -1)
            # Outer circle
            cv2.circle(display, (px, py), cursor_size, cursor_color, 1)
            
            # If hovering, draw object name near cursor
            if hover_obj:
                label = hover_obj.name
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Position label above cursor
                label_x = px - tw // 2
                label_y = py - cursor_size - 10
                
                # Background rectangle
                cv2.rectangle(display, (label_x - 5, label_y - th - 5), 
                            (label_x + tw + 5, label_y + 5), (0, 0, 0), -1)
                cv2.putText(display, label, (label_x, label_y), font, font_scale, 
                           (0, 255, 255), thickness)
        
        self._show_on_canvas(display)
    
    def _sync_from_gesture(self, gesture, action_text="-"):
        """Sync UI from gesture (main thread)."""
        self.scale_var.set(gesture.scale_factor)
        self.rotation_var.set(gesture.rotation_angle)
        self.flip_var.set(gesture.flip_horizontal)
        
        self.gesture_labels["Mode"].config(text="TRANSFORM", foreground="lime")
        self.gesture_labels["Hover"].config(text="-", foreground="gray")
        self.gesture_labels["Selected"].config(
            text=self.selected_object.name if self.selected_object else "-"
        )
        self.gesture_labels["Action"].config(text=action_text)
        
        self._update_display()
    
    def _export_selected(self):
        """Export selected object."""
        if not self.selected_object:
            messagebox.showwarning("Warning", "No object selected")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")]
        )
        if path:
            cv2.imwrite(path, self.selected_object.image)
            self.status_var.set(f"Exported: {os.path.basename(path)}")
    
    def _export_all(self):
        """Export all objects to a folder."""
        if not self.objects:
            messagebox.showwarning("Warning", "No objects to export")
            return
        
        folder = filedialog.askdirectory(title="Select Export Folder")
        if folder:
            for obj in self.objects:
                path = os.path.join(folder, f"{obj.name.replace(' ', '_')}.png")
                cv2.imwrite(path, obj.image)
            self.status_var.set(f"Exported {len(self.objects)} objects to {folder}")
    
    def _save_result(self):
        """Save the result image."""
        if self.original_image is None:
            return
        
        mode = self.transform_target.get()
        
        if mode == "image":
            # Save full image transform
            result = self._apply_full_image_transform()
        else:
            # Apply all object transforms
            result = self.original_image.copy()
            for obj in self.objects:
                if obj.scale != 1.0 or obj.rotation != 0 or obj.flipped:
                    result = self._apply_transform(result, obj)
        
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if path:
            cv2.imwrite(path, result)
            self.status_var.set(f"Saved: {os.path.basename(path)}")
    
    def _reset_all(self):
        """Reset everything."""
        for obj in self.objects:
            obj.scale = 1.0
            obj.rotation = 0.0
            obj.flipped = False
        
        # Reset full image transforms
        self.full_image_scale = 1.0
        self.full_image_rotation = 0.0
        self.full_image_flipped = False
        
        self.scale_var.set(1.0)
        self.rotation_var.set(0.0)
        self.flip_var.set(False)
        
        if self.recognizer:
            self.recognizer.reset()
        
        self._update_display()
        self.status_var.set("Reset complete")
    
    def _on_close(self):
        """Handle close."""
        self._stop_gesture()
        self.root.destroy()
    
    def run(self):
        """Run application."""
        self.root.mainloop()


def main():
    print("\n" + "=" * 60)
    print("  MULTI-OBJECT EXTRACTOR & MANIPULATOR")
    print("  Extract objects from images and manipulate with gestures")
    print("=" * 60 + "\n")
    
    app = MultiObjectManipulator()
    app.run()


if __name__ == "__main__":
    main()
