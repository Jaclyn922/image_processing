import cv2
import numpy as np
from skimage.morphology import skeletonize
from collections import deque
 
def main():
    img_path = "Jacklyn/1.png"
    print(f"Reading image from {img_path}...")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not find or read image file {img_path}")
        return
 
    print("Processing image...")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    img_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
 
    hsv_eq = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)
    # lower_yellow = np.array([18, 60, 70], dtype=np.uint8)
    lower_yellow = np.array([18, 60, 80], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv_eq, lower_yellow, upper_yellow)
 
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv_eq, lower_white, upper_white)
    mask_no_white = cv2.bitwise_and(mask_yellow, cv2.bitwise_not(mask_white))
 
    kernel_open = np.ones((3,3), np.uint8)
    mask_open = cv2.morphologyEx(mask_no_white, cv2.MORPH_OPEN, kernel_open, iterations=1)
 
    kernel_close = np.ones((5,5), np.uint8)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close, iterations=2)
 
    #--------------------------------
    # Visualize the equalized image
    eq_output = "equalized_image.jpg"
    cv2.imwrite(eq_output, img_eq)
    print(f"Equalized image saved to {eq_output}")
    # Visualize intermediate masks
    # Save mask_yellow
    yellow_output = "mask_yellow.jpg"
    cv2.imwrite(yellow_output, mask_yellow)
    print(f"Yellow mask saved to {yellow_output}")
 
    # Save mask_no_white
    no_white_output = "mask_no_white.jpg"
    cv2.imwrite(no_white_output, mask_no_white)
    print(f"No white mask saved to {no_white_output}")
 
    # Save mask_open
    open_output = "mask_open.jpg"
    cv2.imwrite(open_output, mask_open)
    print(f"Open mask saved to {open_output}")
 
    # Save mask_close
    close_output = "mask_close.jpg"
    cv2.imwrite(close_output, mask_close)
    print(f"Close mask saved to {close_output}")
 
    # Optional: Create colored visualizations for better viewing
    def create_colored_mask(mask, color=(0, 0, 255)):
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_mask[mask > 0] = color
        return colored_mask
 
    # Save colored versions
    # Create contour visualizations instead of filled masks
    yellow_contour = img_eq.copy()
    contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(yellow_contour, contours, -1, (0, 0, 255), 2)  # Red color (BGR format)
    cv2.imwrite("mask_yellow_contour.jpg", yellow_contour)
 
    no_white_contour = img_eq.copy()
    contours, _ = cv2.findContours(mask_no_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(no_white_contour, contours, -1, (0, 0, 255), 2)  # Red color (BGR format)
    cv2.imwrite("mask_no_white_contour.jpg", no_white_contour)
 
    open_contour = img_eq.copy()
    contours, _ = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(open_contour, contours, -1, (0, 0, 255), 2)  # Red color (BGR format)
    cv2.imwrite("mask_open_contour.jpg", open_contour)
 
    close_contour = img_eq.copy()
    contours, _ = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(close_contour, contours, -1, (0, 0, 255), 2)  # Red color (BGR format)
    cv2.imwrite("mask_close_contour.jpg", close_contour)
 
    print("Contour visualizations saved")
 
    mask_main = keep_max_connected_component(mask_close)
    # Create a visualization of the mask overlaid on the original image with alpha=0.5
    # Convert mask_main to a 3-channel image (for overlay)
    # Convert mask_main to a 3-channel image with red color for the mask
    mask_rgb = np.zeros_like(img_eq)
    mask_rgb[mask_main > 0] = [0, 0, 255]  # Red color for mask (BGR format)
 
    # Create the overlay image
    alpha = 0.3  # Transparency factor
    overlay_img = cv2.addWeighted(img_eq, 1, mask_rgb, alpha, 0)
 
    # Save the overlay visualization
    overlay_output = "heart_mask_overlay.jpg"
    cv2.imwrite(overlay_output, overlay_img)
    print(f"Mask overlay saved to {overlay_output}")
 
    # Convert the mask to a boolean array where True represents the foreground (heart shape)
    skeleton_bool = (mask_main > 0)
 
    # Apply skeletonization to reduce the shape to a 1-pixel wide representation
    # This creates a thin line (skeleton) that preserves the topological properties of the shape
    skeleton = skeletonize(skeleton_bool)
    # Visualize the skeleton
    # Create a blank image to draw the skeleton on
    skeleton_img = np.zeros_like(img_eq)
 
    # Get coordinates of skeleton pixels
    skeleton_coords = np.where(skeleton)
 
    # Set skeleton pixels to white
    skeleton_img[skeleton_coords] = [255, 255, 255]
 
    # Save the skeleton visualization
    skeleton_output = "heart_skeleton.jpg"
    cv2.imwrite(skeleton_output, skeleton_img)
    print(f"Skeleton visualization saved to {skeleton_output}")
 
    # Optional: Create a version with the skeleton overlaid on the original image
    overlay_skeleton = img_eq.copy()
    overlay_skeleton[skeleton_coords] = [0, 0, 255]  # Red color for skeleton
    cv2.imwrite("heart_skeleton_overlay.jpg", overlay_skeleton)
    print("Skeleton overlay saved to heart_skeleton_overlay.jpg")
 
    # Create a visualization of the skeleton overlaid on the mask_main
    mask_main_rgb = cv2.cvtColor(mask_main, cv2.COLOR_GRAY2BGR)
 
    # Get coordinates of skeleton pixels
    skeleton_coords = np.where(skeleton)
 
    # Create a copy of mask_main_rgb to overlay the skeleton
    mask_skeleton_overlay = mask_main_rgb.copy()
 
    # Set skeleton pixels to red color for better visibility
    mask_skeleton_overlay[skeleton_coords] = [0, 0, 255]  # Red color for skeleton
 
    # Save the skeleton overlaid on mask_main
    mask_skeleton_output = "heart_skeleton_on_mask.jpg"
    cv2.imwrite(mask_skeleton_output, mask_skeleton_overlay)
    print(f"Skeleton overlaid on mask saved to {mask_skeleton_output}")
    #--------------------------------
 
    longest_path = get_longest_path_in_skeleton(skeleton)
    smoothed_path = smooth_path_moving_average(longest_path, window_size=5)
 
    overlay = img_eq.copy()
    draw_smooth_curve(overlay, smoothed_path, color=(0,0,255), thickness=2)
 
    # Save the output file
    output_file = "heart_skeleton_smooth.jpg"
    cv2.imwrite(output_file, overlay)
    print(f"Processing complete. Result saved to {output_file}")
 
    # Skip displaying windows - headless mode
    # cv2.imshow("Final Smooth Skeleton", overlay)
    # while True:
    #     key = cv2.waitKey(50) & 0xFF
    #     if key == 27:
    #         break
    # cv2.destroyAllWindows()
 
def keep_max_connected_component(mask):
    mask_01 = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_01, connectivity=8)
    if num_labels <= 1:
        return mask
    max_area = 0
    max_label = 1
    for lbl in range(1, num_labels):
        area = np.sum(labels == lbl)
        if area > max_area:
            max_area = area
            max_label = lbl
    out = np.zeros_like(mask_01, dtype=np.uint8)
    out[labels == max_label] = 255
    return out
 
def get_longest_path_in_skeleton(skeleton):
    coords = np.column_stack(np.where(skeleton))
    if len(coords) < 2:
        return coords
 
    graph = build_graph_from_skeleton(skeleton, coords)
 
    endpoints = [p for p in graph if len(graph[p]) == 1]
    if len(endpoints) == 0:
        endpoints = [coords[0]]
 
    far_node1, _ = bfs_farthest_node_with_parent(endpoints[0], graph)
    far_node2, parent2 = bfs_farthest_node_with_parent(far_node1, graph)
    longest_path = reconstruct_path(far_node2, parent2)
    return longest_path
 
def build_graph_from_skeleton(skeleton, coords):
    graph = {}
    for (r,c) in coords:
        graph[(r,c)] = []
 
    directions = [(-1,-1), (-1,0), (-1,1),
                  (0,-1),         (0,1),
                  (1,-1),  (1,0),  (1,1)]
    for (r,c) in coords:
        for dr,dc in directions:
            nr, nc = r+dr, c+dc
            if (nr,nc) in graph:
                graph[(r,c)].append((nr,nc))
    return graph
 
def bfs_farthest_node_with_parent(start, graph):
    visited = set([start])
    queue = deque([start])
    parent = {start: None}
    farthest = start
 
    while queue:
        node = queue.popleft()
        farthest = node
        for nxt in graph[node]:
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = node
                queue.append(nxt)
 
    return farthest, parent
 
def reconstruct_path(end_node, parent_map):
    path = []
    cur = end_node
    while cur is not None:
        path.append(cur)
        cur = parent_map[cur]
    path.reverse()
    return path
 
def smooth_path_moving_average(path, window_size=5):
    if len(path) < window_size:
        return path
 
    smoothed = []
    for i in range(len(path)):
        sum_r, sum_c = 0.0, 0.0
        count = 0
        for j in range(i-window_size, i+window_size+1):
            if 0 <= j < len(path):
                sum_r += path[j][0]
                sum_c += path[j][1]
                count += 1
        smoothed.append((sum_r/count, sum_c/count))
 
    return smoothed
 
def draw_smooth_curve(img, path, color=(0,0,255), thickness=2):
    if len(path) < 2:
        return
 
    pts = [(int(round(p[1])), int(round(p[0]))) for p in path]
 
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], color, thickness)
 
if __name__ == "__main__":
    main()
