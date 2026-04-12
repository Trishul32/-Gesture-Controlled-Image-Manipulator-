"""
Generate a 3D-looking image for gesture control demo.
Creates a shaded 3D cube that clearly shows transformations.
"""

import cv2
import numpy as np
import os

def create_3d_cube_image(size=500):
    """
    Create a 3D rendered cube image with shading.
    The cube has different colored faces to show rotation/flip clearly.
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 30  # Dark background
    
    center_x, center_y = size // 2, size // 2
    cube_size = size // 3
    
    # Isometric projection offsets
    offset_x = int(cube_size * 0.5)
    offset_y = int(cube_size * 0.3)
    
    # Define cube vertices (isometric view)
    # Front face
    front_tl = (center_x - cube_size//2, center_y - cube_size//2 + offset_y)
    front_tr = (center_x + cube_size//2, center_y - cube_size//2 + offset_y)
    front_bl = (center_x - cube_size//2, center_y + cube_size//2 + offset_y)
    front_br = (center_x + cube_size//2, center_y + cube_size//2 + offset_y)
    
    # Top face (shifted up and right)
    top_tl = (front_tl[0] + offset_x//2, front_tl[1] - offset_y)
    top_tr = (front_tr[0] + offset_x//2, front_tr[1] - offset_y)
    top_bl = front_tl
    top_br = front_tr
    
    # Right face (shifted right)
    right_tl = front_tr
    right_tr = (front_tr[0] + offset_x//2, front_tr[1] - offset_y)
    right_bl = front_br
    right_br = (front_br[0] + offset_x//2, front_br[1] - offset_y)
    
    # Draw faces (back to front for proper overlap)
    
    # Top face - Light blue (brightest - facing light)
    top_pts = np.array([top_tl, top_tr, top_br, top_bl], np.int32)
    cv2.fillPoly(img, [top_pts], (255, 200, 100))  # Light blue
    cv2.polylines(img, [top_pts], True, (200, 150, 50), 2)
    
    # Right face - Medium blue
    right_pts = np.array([right_tl, right_tr, right_br, right_bl], np.int32)
    cv2.fillPoly(img, [right_pts], (200, 150, 80))  # Medium blue
    cv2.polylines(img, [right_pts], True, (150, 100, 40), 2)
    
    # Front face - Dark blue (shadow side)
    front_pts = np.array([front_tl, front_tr, front_br, front_bl], np.int32)
    cv2.fillPoly(img, [front_pts], (180, 120, 60))  # Darker blue
    cv2.polylines(img, [front_pts], True, (130, 80, 30), 2)
    
    # Add "3D" text on front face
    text_x = center_x - 35
    text_y = center_y + offset_y + 15
    cv2.putText(img, "3D", (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Add directional arrows to show orientation
    # Arrow pointing right (X axis) - Red
    arrow_start = (50, size - 50)
    cv2.arrowedLine(img, arrow_start, (arrow_start[0] + 60, arrow_start[1]), 
                    (0, 0, 255), 3, tipLength=0.3)
    cv2.putText(img, "X", (arrow_start[0] + 65, arrow_start[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Arrow pointing up (Y axis) - Green
    cv2.arrowedLine(img, arrow_start, (arrow_start[0], arrow_start[1] - 60),
                    (0, 255, 0), 3, tipLength=0.3)
    cv2.putText(img, "Y", (arrow_start[0] - 5, arrow_start[1] - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add corner markers for orientation reference
    marker_size = 25
    # Top-left: Red
    cv2.rectangle(img, (10, 10), (10 + marker_size, 10 + marker_size), (0, 0, 255), -1)
    cv2.putText(img, "TL", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Top-right: Green
    cv2.rectangle(img, (size - 35, 10), (size - 10, 10 + marker_size), (0, 255, 0), -1)
    cv2.putText(img, "TR", (size - 33, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Bottom-left: Blue
    cv2.rectangle(img, (10, size - 35), (10 + marker_size, size - 10), (255, 0, 0), -1)
    cv2.putText(img, "BL", (12, size - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Bottom-right: Yellow
    cv2.rectangle(img, (size - 35, size - 35), (size - 10, size - 10), (0, 255, 255), -1)
    cv2.putText(img, "BR", (size - 33, size - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return img


def create_3d_sphere_image(size=500):
    """
    Create a 3D shaded sphere image.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Gradient background
    for y in range(size):
        ratio = y / size
        color = int(20 + ratio * 30)
        img[y, :] = (color, color, color + 10)
    
    center = (size // 2, size // 2)
    radius = size // 3
    
    # Draw sphere with gradient shading
    for y in range(size):
        for x in range(size):
            dx = x - center[0]
            dy = y - center[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist <= radius:
                # Calculate 3D position on sphere
                z = np.sqrt(radius*radius - dist*dist)
                
                # Normal vector
                nx, ny, nz = dx/radius, dy/radius, z/radius
                
                # Light direction (from top-left)
                lx, ly, lz = -0.5, -0.5, 0.7
                l_len = np.sqrt(lx*lx + ly*ly + lz*lz)
                lx, ly, lz = lx/l_len, ly/l_len, lz/l_len
                
                # Diffuse lighting
                diffuse = max(0, nx*lx + ny*ly + nz*lz)
                
                # Specular highlight
                rx, ry, rz = 2*nz*nx - lx, 2*nz*ny - ly, 2*nz*nz - lz
                spec = max(0, rz) ** 32
                
                # Color (orange-red gradient)
                base_r = 220
                base_g = 100
                base_b = 50
                
                r = int(min(255, base_r * (0.2 + 0.8 * diffuse) + 255 * spec))
                g = int(min(255, base_g * (0.2 + 0.8 * diffuse) + 255 * spec))
                b = int(min(255, base_b * (0.2 + 0.8 * diffuse) + 255 * spec))
                
                img[y, x] = (b, g, r)
    
    # Add shadow
    shadow_center = (center[0] + 20, center[1] + radius + 30)
    shadow_axes = (radius - 10, radius // 4)
    overlay = img.copy()
    cv2.ellipse(overlay, shadow_center, shadow_axes, 0, 0, 360, (10, 10, 10), -1)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    
    # Add orientation markers
    marker_size = 25
    cv2.rectangle(img, (10, 10), (35, 35), (0, 0, 255), -1)  # TL Red
    cv2.rectangle(img, (size-35, 10), (size-10, 35), (0, 255, 0), -1)  # TR Green
    cv2.rectangle(img, (10, size-35), (35, size-10), (255, 0, 0), -1)  # BL Blue
    cv2.rectangle(img, (size-35, size-35), (size-10, size-10), (0, 255, 255), -1)  # BR Yellow
    
    # Labels
    cv2.putText(img, "TL", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(img, "TR", (size-33, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, "BL", (12, size-17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(img, "BR", (size-33, size-17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return img


def create_3d_pyramid_image(size=500):
    """
    Create a 3D pyramid image with shaded faces.
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 35  # Dark gray background
    
    center_x = size // 2
    base_y = int(size * 0.75)
    apex_y = int(size * 0.15)
    base_half = size // 3
    
    # Pyramid vertices
    apex = (center_x, apex_y)
    base_left = (center_x - base_half, base_y)
    base_right = (center_x + base_half, base_y)
    base_back = (center_x + base_half // 2, base_y - base_half // 2)
    
    # Draw back-right face (darkest)
    back_face = np.array([apex, base_right, base_back], np.int32)
    cv2.fillPoly(img, [back_face], (80, 60, 40))
    cv2.polylines(img, [back_face], True, (60, 40, 20), 2)
    
    # Draw base (partial, visible part)
    base_visible = np.array([base_left, base_right, base_back], np.int32)
    cv2.fillPoly(img, [base_visible], (60, 50, 35))
    cv2.polylines(img, [base_visible], True, (40, 30, 15), 2)
    
    # Draw front-left face (brighter, facing light)
    front_left = np.array([apex, base_left, base_back], np.int32)
    cv2.fillPoly(img, [front_left], (180, 140, 80))
    cv2.polylines(img, [front_left], True, (140, 100, 50), 2)
    
    # Draw front face (brightest)
    front_face = np.array([apex, base_left, base_right], np.int32)
    cv2.fillPoly(img, [front_face], (220, 180, 100))
    cv2.polylines(img, [front_face], True, (180, 140, 60), 2)
    
    # Add "PYRAMID" text
    cv2.putText(img, "PYRAMID", (center_x - 70, base_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    
    # Corner markers
    cv2.rectangle(img, (10, 10), (35, 35), (0, 0, 255), -1)
    cv2.rectangle(img, (size-35, 10), (size-10, 35), (0, 255, 0), -1)
    cv2.rectangle(img, (10, size-35), (35, size-10), (255, 0, 0), -1)
    cv2.rectangle(img, (size-35, size-35), (size-10, size-10), (0, 255, 255), -1)
    
    cv2.putText(img, "TL", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(img, "TR", (size-33, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, "BL", (12, size-17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(img, "BR", (size-33, size-17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return img


def create_combined_3d_image(size=500):
    """
    Create a combined image with multiple 3D objects.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Gradient background (dark blue to black)
    for y in range(size):
        ratio = y / size
        b = int(60 - ratio * 40)
        g = int(30 - ratio * 20)
        img[y, :] = (b, g, 10)
    
    # Grid floor
    floor_y_start = int(size * 0.6)
    for i in range(10):
        # Horizontal lines (perspective)
        y = floor_y_start + i * 15
        if y < size:
            alpha = 1.0 - (i / 10) * 0.7
            color = (int(80 * alpha), int(80 * alpha), int(100 * alpha))
            cv2.line(img, (0, y), (size, y), color, 1)
    
    for i in range(15):
        # Vertical lines (converging to horizon)
        x_bottom = i * (size // 14)
        x_top = size//2 + (x_bottom - size//2) // 3
        alpha = 0.5
        color = (int(80 * alpha), int(80 * alpha), int(100 * alpha))
        cv2.line(img, (x_bottom, size), (x_top, floor_y_start), color, 1)
    
    # Draw 3D cube
    cube_center = (size // 3, size // 2)
    cube_size = 80
    
    # Cube faces
    # Front
    front = np.array([
        [cube_center[0] - cube_size//2, cube_center[1] - cube_size//2],
        [cube_center[0] + cube_size//2, cube_center[1] - cube_size//2],
        [cube_center[0] + cube_size//2, cube_center[1] + cube_size//2],
        [cube_center[0] - cube_size//2, cube_center[1] + cube_size//2]
    ], np.int32)
    
    offset = 30
    # Top
    top = np.array([
        [cube_center[0] - cube_size//2 + offset, cube_center[1] - cube_size//2 - offset//2],
        [cube_center[0] + cube_size//2 + offset, cube_center[1] - cube_size//2 - offset//2],
        [cube_center[0] + cube_size//2, cube_center[1] - cube_size//2],
        [cube_center[0] - cube_size//2, cube_center[1] - cube_size//2]
    ], np.int32)
    
    # Right
    right = np.array([
        [cube_center[0] + cube_size//2, cube_center[1] - cube_size//2],
        [cube_center[0] + cube_size//2 + offset, cube_center[1] - cube_size//2 - offset//2],
        [cube_center[0] + cube_size//2 + offset, cube_center[1] + cube_size//2 - offset//2],
        [cube_center[0] + cube_size//2, cube_center[1] + cube_size//2]
    ], np.int32)
    
    cv2.fillPoly(img, [top], (255, 180, 100))
    cv2.fillPoly(img, [right], (200, 130, 70))
    cv2.fillPoly(img, [front], (160, 100, 50))
    cv2.polylines(img, [front, top, right], True, (100, 60, 30), 2)
    
    # Draw 3D sphere
    sphere_center = (int(size * 0.7), int(size * 0.45))
    sphere_radius = 60
    
    for y in range(sphere_center[1] - sphere_radius, sphere_center[1] + sphere_radius):
        for x in range(sphere_center[0] - sphere_radius, sphere_center[0] + sphere_radius):
            if 0 <= x < size and 0 <= y < size:
                dx = x - sphere_center[0]
                dy = y - sphere_center[1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist <= sphere_radius:
                    z = np.sqrt(sphere_radius**2 - dist**2)
                    nz = z / sphere_radius
                    nx = dx / sphere_radius
                    ny = dy / sphere_radius
                    
                    # Lighting
                    light = max(0, -0.5*nx - 0.5*ny + 0.7*nz)
                    spec = max(0, nz) ** 16
                    
                    r = int(min(255, 100 * (0.3 + 0.7 * light) + 200 * spec))
                    g = int(min(255, 200 * (0.3 + 0.7 * light) + 200 * spec))
                    b = int(min(255, 100 * (0.3 + 0.7 * light) + 200 * spec))
                    
                    img[y, x] = (b, g, r)
    
    # Title
    cv2.putText(img, "3D SCENE", (size//2 - 80, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 220), 2)
    
    # Add orientation markers
    cv2.rectangle(img, (10, 10), (35, 35), (0, 0, 255), -1)
    cv2.rectangle(img, (size-35, 10), (size-10, 35), (0, 255, 0), -1)
    cv2.rectangle(img, (10, size-35), (35, size-10), (255, 0, 0), -1)
    cv2.rectangle(img, (size-35, size-35), (size-10, size-10), (0, 255, 255), -1)
    
    cv2.putText(img, "TL", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(img, "TR", (size-33, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, "BL", (12, size-17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(img, "BR", (size-33, size-17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return img


def main():
    """Generate and save all 3D images."""
    os.makedirs("assets", exist_ok=True)
    
    print("Generating 3D images...")
    
    # Generate images
    cube = create_3d_cube_image(500)
    sphere = create_3d_sphere_image(500)
    pyramid = create_3d_pyramid_image(500)
    scene = create_combined_3d_image(500)
    
    # Save images
    cv2.imwrite("assets/3d_cube.png", cube)
    print("  Saved: assets/3d_cube.png")
    
    cv2.imwrite("assets/3d_sphere.png", sphere)
    print("  Saved: assets/3d_sphere.png")
    
    cv2.imwrite("assets/3d_pyramid.png", pyramid)
    print("  Saved: assets/3d_pyramid.png")
    
    cv2.imwrite("assets/3d_scene.png", scene)
    print("  Saved: assets/3d_scene.png")
    
    # Set default
    cv2.imwrite("assets/sample_image.png", scene)
    print("  Saved: assets/sample_image.png (default)")
    
    print("\nDone! Images saved to assets/")
    print("\nPreview images? (press any key to close each)")
    
    cv2.imshow("3D Cube", cube)
    cv2.waitKey(0)
    cv2.imshow("3D Sphere", sphere)
    cv2.waitKey(0)
    cv2.imshow("3D Pyramid", pyramid)
    cv2.waitKey(0)
    cv2.imshow("3D Scene", scene)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
