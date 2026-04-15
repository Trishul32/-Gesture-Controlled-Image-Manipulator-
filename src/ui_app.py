"""
Gesture-Controlled Object Manipulator - UI Application
Upload images, select objects, and manipulate them with hand gestures.

Features:
    - Upload and load images
    - Click to select objects
    - Multiple selection methods (smart, flood fill, contour)
    - Gesture-based transformations on selected objects
    - Save results

Usage:
    python src/ui_app.py
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from object_selector import ObjectSelector, SelectedObject
from hand_detector import HandDetector, WebcamCapture
from gesture_recognizer import GestureRecognizer, GestureResult
from image_transformer import ImageTransformer


class GestureObjectManipulator:
    """
    Main UI Application for gesture-controlled object manipulation.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.root = tk.Tk()
        self.root.title("Gesture-Controlled Object Manipulator")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2b2b2b')
        
        # State
        self.current_image = None
        self.original_image = None
        self.selector = None
        self.selected_object: SelectedObject = None
        self.transformed_object = None
        
        # Gesture control state
        self.gesture_active = False
        self.detector = None
        self.recognizer = None
        self.webcam = None
        self.gesture_thread = None
        self.current_gesture = GestureResult()
        
        # Selection method
        self.selection_method = tk.StringVar(value="smart")
        self.tolerance = tk.IntVar(value=20)
        
        # Transformation state
        self.scale = 1.0
        self.rotation = 0.0
        self.translation = (0, 0)
        self.flipped = False
        
        # Build UI
        self._build_ui()
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _build_ui(self):
        """Build the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure style
        style = ttk.Style()
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TButton', padding=5)
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        
        # Left panel - Controls
        control_frame = ttk.Frame(main_frame, width=250)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        self._build_control_panel(control_frame)
        
        # Center - Image display
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self._build_image_panel(image_frame)
        
        # Right panel - Webcam/Gesture
        gesture_frame = ttk.Frame(main_frame, width=320)
        gesture_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        gesture_frame.pack_propagate(False)
        
        self._build_gesture_panel(gesture_frame)
    
    def _build_control_panel(self, parent):
        """Build control panel with buttons and settings."""
        # Title
        ttk.Label(parent, text="CONTROLS", style='Header.TLabel').pack(pady=(0, 15))
        
        # File operations
        file_frame = ttk.LabelFrame(parent, text="File")
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="📁 Upload Image", 
                  command=self._upload_image).pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(file_frame, text="💾 Save Result", 
                  command=self._save_result).pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(file_frame, text="🔄 Reset", 
                  command=self._reset_all).pack(fill=tk.X, padx=5, pady=3)
        
        # Selection method
        sel_frame = ttk.LabelFrame(parent, text="Selection Method")
        sel_frame.pack(fill=tk.X, pady=10)
        
        methods = [
            ("Smart (Auto)", "smart"),
            ("Flood Fill", "flood"),
            ("Contour", "contour"),
            ("Color", "color")
        ]
        for text, value in methods:
            ttk.Radiobutton(sel_frame, text=text, value=value,
                           variable=self.selection_method).pack(anchor=tk.W, padx=5)
        
        # Tolerance slider
        ttk.Label(sel_frame, text="Tolerance:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        tolerance_scale = ttk.Scale(sel_frame, from_=5, to=100, 
                                   variable=self.tolerance, orient=tk.HORIZONTAL)
        tolerance_scale.pack(fill=tk.X, padx=5)
        
        # Object info
        self.info_frame = ttk.LabelFrame(parent, text="Selected Object")
        self.info_frame.pack(fill=tk.X, pady=10)
        
        self.info_label = ttk.Label(self.info_frame, text="No object selected")
        self.info_label.pack(padx=5, pady=5)
        
        # Transformation controls
        trans_frame = ttk.LabelFrame(parent, text="Manual Transform")
        trans_frame.pack(fill=tk.X, pady=10)
        
        # Scale
        ttk.Label(trans_frame, text="Scale:").pack(anchor=tk.W, padx=5)
        self.scale_var = tk.DoubleVar(value=1.0)
        scale_slider = ttk.Scale(trans_frame, from_=0.1, to=3.0, 
                                variable=self.scale_var, orient=tk.HORIZONTAL,
                                command=self._on_manual_transform)
        scale_slider.pack(fill=tk.X, padx=5)
        
        # Rotation
        ttk.Label(trans_frame, text="Rotation:").pack(anchor=tk.W, padx=5)
        self.rotation_var = tk.DoubleVar(value=0.0)
        rotation_slider = ttk.Scale(trans_frame, from_=-180, to=180, 
                                   variable=self.rotation_var, orient=tk.HORIZONTAL,
                                   command=self._on_manual_transform)
        rotation_slider.pack(fill=tk.X, padx=5)
        
        # Flip
        self.flip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(trans_frame, text="Flip Horizontal", 
                       variable=self.flip_var,
                       command=self._on_manual_transform).pack(anchor=tk.W, padx=5, pady=5)
        
        # Instructions
        inst_frame = ttk.LabelFrame(parent, text="Instructions")
        inst_frame.pack(fill=tk.X, pady=10)
        
        instructions = """
1. Upload an image
2. Click on an object to select
3. Enable gesture control
4. Use hand gestures:
   • Pinch → Zoom
   • Tilt → Rotate
   • Move → Pan
   • Fist → Flip
        """
        ttk.Label(inst_frame, text=instructions.strip(), 
                 wraplength=220, justify=tk.LEFT).pack(padx=5, pady=5)
    
    def _build_image_panel(self, parent):
        """Build image display panel."""
        # Title
        ttk.Label(parent, text="IMAGE VIEW", style='Header.TLabel').pack(pady=(0, 10))
        
        # Canvas for image display
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='#1e1e1e', 
                               highlightthickness=1, highlightbackground='#555')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        
        # Status bar
        self.status_var = tk.StringVar(value="Click 'Upload Image' to start")
        status_bar = ttk.Label(parent, textvariable=self.status_var)
        status_bar.pack(fill=tk.X, pady=(5, 0))
    
    def _build_gesture_panel(self, parent):
        """Build gesture control panel with webcam view."""
        # Title
        ttk.Label(parent, text="GESTURE CONTROL", style='Header.TLabel').pack(pady=(0, 10))
        
        # Webcam view
        self.webcam_label = ttk.Label(parent)
        self.webcam_label.pack(pady=5)
        
        # Placeholder image
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Webcam Preview", (60, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        self._update_webcam_display(placeholder)
        
        # Control buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.gesture_btn_text = tk.StringVar(value="▶ Start Gesture Control")
        self.gesture_btn = ttk.Button(btn_frame, textvariable=self.gesture_btn_text,
                                     command=self._toggle_gesture_control)
        self.gesture_btn.pack(fill=tk.X, padx=5)
        
        ttk.Button(btn_frame, text="🔄 Reset Calibration",
                  command=self._reset_calibration).pack(fill=tk.X, padx=5, pady=5)
        
        # Gesture info display
        gesture_info_frame = ttk.LabelFrame(parent, text="Gesture Values")
        gesture_info_frame.pack(fill=tk.X, pady=10)
        
        self.gesture_info_labels = {}
        for name in ["Hand", "Scale", "Rotation", "Pan X", "Pan Y", "Flip"]:
            row = ttk.Frame(gesture_info_frame)
            row.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(row, text=f"{name}:", width=10).pack(side=tk.LEFT)
            label = ttk.Label(row, text="-", width=15)
            label.pack(side=tk.LEFT)
            self.gesture_info_labels[name] = label
        
        # Transform mode
        mode_frame = ttk.LabelFrame(parent, text="Transform Mode")
        mode_frame.pack(fill=tk.X, pady=10)
        
        self.transform_mode = tk.StringVar(value="object")
        ttk.Radiobutton(mode_frame, text="Selected Object Only", 
                       value="object", variable=self.transform_mode).pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(mode_frame, text="Entire Image", 
                       value="image", variable=self.transform_mode).pack(anchor=tk.W, padx=5)
    
    def _upload_image(self):
        """Open file dialog to upload image."""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=filetypes
        )
        
        if filepath:
            self._load_image(filepath)
    
    def _load_image(self, filepath: str):
        """Load image from file."""
        try:
            # Load with OpenCV
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("Could not load image")
            
            # Store original
            self.original_image = image.copy()
            self.current_image = image.copy()
            
            # Create selector
            self.selector = ObjectSelector(image)
            
            # Clear selection
            self.selected_object = None
            self._reset_transforms()
            
            # Update display
            self._update_image_display()
            self.status_var.set(f"Loaded: {os.path.basename(filepath)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def _on_canvas_click(self, event):
        """Handle left click on canvas to select object."""
        if self.current_image is None or self.selector is None:
            return
        
        # Convert canvas coordinates to image coordinates
        x, y = self._canvas_to_image_coords(event.x, event.y)
        if x is None:
            return
        
        # Select object
        method = self.selection_method.get()
        tolerance = self.tolerance.get()
        
        if method == "color":
            self.selected_object = self.selector.select_by_color(x, y, tolerance)
        else:
            self.selected_object = self.selector.select_at_point(x, y, tolerance, method)
        
        if self.selected_object:
            self._reset_transforms()
            self._update_object_info()
            self.status_var.set(f"Selected object at ({x}, {y})")
        else:
            self.status_var.set("No object found at click position")
        
        self._update_image_display()
    
    def _on_canvas_right_click(self, event):
        """Handle right click to clear selection."""
        self.selected_object = None
        self._reset_transforms()
        self._update_object_info()
        self._update_image_display()
        self.status_var.set("Selection cleared")
    
    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        if self.current_image is not None:
            self._update_image_display()
    
    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates."""
        if self.current_image is None:
            return None, None
        
        # Get canvas size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        # Get image size
        img_h, img_w = self.current_image.shape[:2]
        
        # Calculate scale and offset
        scale = min(canvas_w / img_w, canvas_h / img_h)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        
        offset_x = (canvas_w - scaled_w) // 2
        offset_y = (canvas_h - scaled_h) // 2
        
        # Convert coordinates
        img_x = int((canvas_x - offset_x) / scale)
        img_y = int((canvas_y - offset_y) / scale)
        
        # Check bounds
        if 0 <= img_x < img_w and 0 <= img_y < img_h:
            return img_x, img_y
        return None, None
    
    def _update_image_display(self):
        """Update the image display on canvas."""
        if self.current_image is None:
            return
        
        # Start with original
        display = self.original_image.copy()
        
        # Apply transformations to selected object or full image
        if self.selected_object and self.transform_mode.get() == "object":
            display = self._apply_object_transform(display)
        elif self.transform_mode.get() == "image":
            display = self._apply_full_transform(display)
        
        # Draw selection highlight
        if self.selected_object:
            overlay = display.copy()
            cv2.drawContours(overlay, [self.selected_object.contour], -1, (0, 255, 0), -1)
            display = cv2.addWeighted(overlay, 0.2, display, 0.8, 0)
            cv2.drawContours(display, [self.selected_object.contour], -1, (0, 255, 0), 2)
            
            # Draw bounding box
            x, y, w, h = self.selected_object.bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Convert to Tk image
        self._display_on_canvas(display)
    
    def _apply_object_transform(self, image):
        """Apply transformation to selected object only."""
        if not self.selected_object:
            return image
        
        result = image.copy()
        
        # Get object image and mask
        obj_img = self.selected_object.image.copy()
        mask = self.selected_object.mask.copy()
        x, y, w, h = self.selected_object.bbox
        
        # Create transformer for the object
        transformer = ImageTransformer()
        transformer.set_image(obj_img)
        
        # Apply transforms
        transformed = transformer.apply_all(
            scale=self.scale,
            rotation=self.rotation,
            translation=self.translation,
            flip_horizontal=self.flipped
        )
        
        # Resize back if needed
        new_h, new_w = transformed.shape[:2]
        
        # Calculate placement position (centered)
        cx, cy = self.selected_object.center
        new_x = cx - new_w // 2 + int(self.translation[0])
        new_y = cy - new_h // 2 + int(self.translation[1])
        
        # Clip to image bounds
        src_x1 = max(0, -new_x)
        src_y1 = max(0, -new_y)
        dst_x1 = max(0, new_x)
        dst_y1 = max(0, new_y)
        
        src_x2 = min(new_w, result.shape[1] - new_x)
        src_y2 = min(new_h, result.shape[0] - new_y)
        dst_x2 = min(result.shape[1], new_x + new_w)
        dst_y2 = min(result.shape[0], new_y + new_h)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            # Erase original object area
            cv2.drawContours(result, [self.selected_object.contour], -1, 
                           tuple(int(x) for x in cv2.mean(image)[:3]), -1)
            
            # Place transformed object
            try:
                result[dst_y1:dst_y2, dst_x1:dst_x2] = transformed[src_y1:src_y2, src_x1:src_x2]
            except:
                pass
        
        return result
    
    def _apply_full_transform(self, image):
        """Apply transformation to entire image."""
        transformer = ImageTransformer()
        transformer.set_image(image)
        
        h, w = image.shape[:2]
        result = transformer.apply_all(
            scale=self.scale,
            rotation=self.rotation,
            translation=self.translation,
            flip_horizontal=self.flipped
        )
        
        # Resize to original dimensions
        return cv2.resize(result, (w, h))
    
    def _display_on_canvas(self, image):
        """Display OpenCV image on Tk canvas."""
        # Get canvas size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            return
        
        # Calculate scale to fit
        img_h, img_w = image.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h) * 0.95
        
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image then to Tk
        pil_image = Image.fromarray(rgb)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display
        self.canvas.delete("all")
        x = canvas_w // 2
        y = canvas_h // 2
        self.canvas.create_image(x, y, image=self.tk_image, anchor=tk.CENTER)
    
    def _update_webcam_display(self, frame):
        """Update webcam preview."""
        # Resize to fit
        frame = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        self.webcam_tk_image = ImageTk.PhotoImage(pil_image)
        self.webcam_label.configure(image=self.webcam_tk_image)
    
    def _update_object_info(self):
        """Update selected object info display."""
        if self.selected_object:
            x, y, w, h = self.selected_object.bbox
            info = f"Size: {w}x{h}\nArea: {self.selected_object.area} px\nCenter: {self.selected_object.center}"
            self.info_label.config(text=info)
        else:
            self.info_label.config(text="No object selected")
    
    def _on_manual_transform(self, *args):
        """Handle manual transformation slider changes."""
        self.scale = self.scale_var.get()
        self.rotation = self.rotation_var.get()
        self.flipped = self.flip_var.get()
        self._update_image_display()
    
    def _reset_transforms(self):
        """Reset all transformations."""
        self.scale = 1.0
        self.rotation = 0.0
        self.translation = (0, 0)
        self.flipped = False
        
        self.scale_var.set(1.0)
        self.rotation_var.set(0.0)
        self.flip_var.set(False)
    
    def _toggle_gesture_control(self):
        """Toggle gesture control on/off."""
        if self.gesture_active:
            self._stop_gesture_control()
        else:
            self._start_gesture_control()
    
    def _start_gesture_control(self):
        """Start gesture control."""
        try:
            # Initialize components
            self.detector = HandDetector(max_hands=1)
            self.recognizer = GestureRecognizer()
            self.webcam = WebcamCapture()
            
            self.gesture_active = True
            self.gesture_btn_text.set("⏹ Stop Gesture Control")
            
            # Start processing thread
            self.gesture_thread = threading.Thread(target=self._gesture_loop, daemon=True)
            self.gesture_thread.start()
            
            self.status_var.set("Gesture control active - show your hand!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start gesture control: {e}")
            self._stop_gesture_control()
    
    def _stop_gesture_control(self):
        """Stop gesture control."""
        self.gesture_active = False
        self.gesture_btn_text.set("▶ Start Gesture Control")
        
        # Clean up
        if self.webcam:
            self.webcam.release()
            self.webcam = None
        if self.detector:
            self.detector.release()
            self.detector = None
        
        # Show placeholder
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Webcam Off", (90, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        self._update_webcam_display(placeholder)
        
        self.status_var.set("Gesture control stopped")
    
    def _gesture_loop(self):
        """Main gesture processing loop (runs in thread)."""
        while self.gesture_active and self.webcam:
            try:
                ret, frame = self.webcam.read()
                if not ret:
                    continue
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                
                # Detect hand
                landmarks = self.detector.detect(frame)
                
                # Recognize gesture
                if landmarks:
                    gesture = self.recognizer.recognize(
                        landmarks,
                        self.webcam.width,
                        self.webcam.height
                    )
                    self.current_gesture = gesture
                    
                    # Update transforms
                    self.scale = gesture.scale_factor
                    self.rotation = gesture.rotation_angle
                    self.translation = gesture.translation
                    self.flipped = gesture.flip_horizontal
                    
                    # Update sliders (thread-safe)
                    self.root.after(0, self._update_from_gesture)
                
                # Draw landmarks
                self.detector.draw_landmarks(frame, landmarks)
                
                # Update webcam display
                self.root.after(0, lambda f=frame.copy(): self._update_webcam_display(f))
                
                # Update gesture info
                self.root.after(0, self._update_gesture_info)
                
            except Exception as e:
                print(f"Gesture loop error: {e}")
    
    def _update_from_gesture(self):
        """Update UI from gesture values (called from main thread)."""
        self.scale_var.set(self.scale)
        self.rotation_var.set(self.rotation)
        self.flip_var.set(self.flipped)
        self._update_image_display()
    
    def _update_gesture_info(self):
        """Update gesture info display."""
        g = self.current_gesture
        
        self.gesture_info_labels["Hand"].config(
            text="Detected" if g.hand_detected else "None",
            foreground="green" if g.hand_detected else "red"
        )
        self.gesture_info_labels["Scale"].config(text=f"{g.scale_factor:.2f}x")
        self.gesture_info_labels["Rotation"].config(text=f"{g.rotation_angle:.1f}°")
        self.gesture_info_labels["Pan X"].config(text=f"{g.translation[0]}")
        self.gesture_info_labels["Pan Y"].config(text=f"{g.translation[1]}")
        self.gesture_info_labels["Flip"].config(
            text="ON" if g.flip_horizontal else "OFF",
            foreground="yellow" if g.flip_horizontal else "gray"
        )
    
    def _reset_calibration(self):
        """Reset gesture calibration."""
        if self.recognizer:
            self.recognizer.reset_calibration()
            self.status_var.set("Gesture calibration reset")
    
    def _save_result(self):
        """Save the current result image."""
        if self.original_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
        
        # Apply current transforms
        if self.selected_object and self.transform_mode.get() == "object":
            result = self._apply_object_transform(self.original_image.copy())
        else:
            result = self._apply_full_transform(self.original_image.copy())
        
        # Ask for save location
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            cv2.imwrite(filepath, result)
            self.status_var.set(f"Saved: {os.path.basename(filepath)}")
    
    def _reset_all(self):
        """Reset everything."""
        self.selected_object = None
        self._reset_transforms()
        
        if self.recognizer:
            self.recognizer.reset()
        
        self._update_object_info()
        self._update_image_display()
        self.status_var.set("Reset complete")
    
    def _on_close(self):
        """Handle window close."""
        self._stop_gesture_control()
        self.root.destroy()
    
    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    """Entry point."""
    print("\n" + "=" * 60)
    print("  GESTURE-CONTROLLED OBJECT MANIPULATOR")
    print("  Upload images, select objects, control with gestures")
    print("=" * 60 + "\n")
    
    app = GestureObjectManipulator()
    app.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
