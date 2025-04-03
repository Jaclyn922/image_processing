import cv2
import numpy as np
from skimage.morphology import skeletonize
from collections import deque

def main():
    img_path = "/Users/Xining/Desktop/Jacklyn_code1.jpg"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    img_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    hsv_eq = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)
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

    mask_main = keep_max_connected_component(mask_close)

    skeleton_bool = (mask_main > 0)
    skeleton = skeletonize(skeleton_bool)

    longest_path = get_longest_path_in_skeleton(skeleton)
    smoothed_path = smooth_path_moving_average(longest_path, window_size=5)

    overlay = img_eq.copy()
    draw_smooth_curve(overlay, smoothed_path, color=(0,0,255), thickness=2)

    cv2.imshow("Final Smooth Skeleton", overlay)
    cv2.imwrite("heart_skeleton_smooth.jpg", overlay)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()

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
