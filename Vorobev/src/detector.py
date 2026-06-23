import cv2
import numpy as np
import math

def process_with_params(image, sheet_mask, params):
    bright_image = cv2.convertScaleAbs(image, alpha=params['alpha'])
    gray_bright = cv2.cvtColor(bright_image, cv2.COLOR_BGR2GRAY)

    k_size = params['tophat']
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    tophat = cv2.morphologyEx(gray_bright, cv2.MORPH_TOPHAT, tophat_kernel)

    _, thresh = cv2.threshold(tophat, params['thresh'], 255, cv2.THRESH_BINARY)
    
    noise_kernel = np.ones((2, 2), np.uint8)
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel)
    
    final_thresh = cv2.bitwise_and(clean_thresh, clean_thresh, mask=sheet_mask)

    contours, _ = cv2.findContours(final_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    markers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 400:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 4 or h < 4:
                continue
                
            aspect_ratio = float(w) / h
            if 0.4 < aspect_ratio < 5:
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area) / hull_area
                    if 0.15 < solidity < 0.8:
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]
                            markers.append({"x": cx, "y": cy, "box": (x, y, w, h), "area": area}) 
    return markers, final_thresh, bright_image


def find_cross_rough_center(gray_img):
    h, w = gray_img.shape[:2]
    steps = []
    steps.append((cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), "0. Исходник (gray)"))

    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (101, 101))
    background = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, bg_kernel)
    normalized = cv2.subtract(gray_img, background)
    steps.append((normalized.copy(), "1. Нормализация фона"))

    vert_size = max(15, h // 10)
    horiz_size = max(15, w // 10)

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))

    vert_lines = cv2.morphologyEx(normalized, cv2.MORPH_OPEN, vert_kernel)
    horiz_lines = cv2.morphologyEx(normalized, cv2.MORPH_OPEN, horiz_kernel)

    steps.append((vert_lines.copy(), "2. Вертикальные линии"))
    steps.append((horiz_lines.copy(), "3. Горизонтальные линии"))

    vert_f = vert_lines.astype(np.float32)
    horiz_f = horiz_lines.astype(np.float32)

    if vert_f.max() > 0: vert_f /= vert_f.max()
    if horiz_f.max() > 0: horiz_f /= horiz_f.max()

    intersection_map = vert_f * horiz_f
    intersection_map = cv2.GaussianBlur(intersection_map, (31, 31), 0)

    if intersection_map.max() > 0:
        heat_vis = (intersection_map / intersection_map.max() * 255).astype(np.uint8)
        heat_colored = cv2.applyColorMap(heat_vis, cv2.COLORMAP_JET)
        steps.append((heat_colored, "4. Карта пересечений"))
    else:
        steps.append((np.zeros((h, w, 3), dtype=np.uint8), "4. Карта пересечений (пусто)"))

    if intersection_map.max() < 1e-8:
        return None, None, None, normalized, steps

    peak_candidates = _find_all_peaks(intersection_map, min_distance=30)

    if not peak_candidates:
        _, _, _, max_loc = cv2.minMaxLoc(intersection_map)
        peak_candidates = [{'x': float(max_loc[0]), 'y': float(max_loc[1]),
                            'strength': intersection_map[max_loc[1], max_loc[0]]}]

    best_cx, best_cy, best_score = _select_best_peak_by_symmetry(
        peak_candidates, normalized, vert_lines, horiz_lines
    )

    result_vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    for cand in peak_candidates:
        cv2.circle(result_vis, (int(cand['x']), int(cand['y'])), 8, (0, 255, 0), 1)
    cv2.circle(result_vis, (int(best_cx), int(best_cy)), 15, (0, 0, 255), 2)
    cv2.drawMarker(result_vis, (int(best_cx), int(best_cy)),
                   (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    steps.append((result_vis,
                  f"5. Грубый центр ({int(best_cx)}, {int(best_cy)}) из {len(peak_candidates)} пиков"))

    box_size = 40
    box = (int(best_cx) - box_size // 2, int(best_cy) - box_size // 2, box_size, box_size)

    return best_cx, best_cy, box, normalized, steps


def _find_all_peaks(score_map, min_distance=30):
    peaks = []
    temp_map = score_map.copy()

    threshold = temp_map.max() * 0.30

    while True:
        _, max_val, _, max_loc = cv2.minMaxLoc(temp_map)
        if max_val < threshold:
            break

        peaks.append({
            'x': float(max_loc[0]),
            'y': float(max_loc[1]),
            'strength': float(max_val)
        })

        cv2.circle(temp_map, max_loc, min_distance, 0, -1)

    return peaks


def _select_best_peak_by_symmetry(peaks, normalized_img, vert_map, horiz_map):
 
    h, w = normalized_img.shape[:2]
    scan_len = min(h, w) // 4  

    best_peak = None
    best_score = -1.0

    for peak in peaks:
        cx, cy = int(peak['x']), int(peak['y'])

        y_start_up = max(0, cy - scan_len)
        prof_up = normalized_img[y_start_up:cy, cx].astype(np.float32)[::-1]

        y_end_down = min(h, cy + scan_len)
        prof_down = normalized_img[cy:y_end_down, cx].astype(np.float32)

        x_start_left = max(0, cx - scan_len)
        prof_left = normalized_img[cy, x_start_left:cx].astype(np.float32)[::-1]

        x_end_right = min(w, cx + scan_len)
        prof_right = normalized_img[cy, cx:x_end_right].astype(np.float32)

        if any(len(p) < 5 for p in [prof_up, prof_down, prof_left, prof_right]):
            continue

        def ray_energy(profile, threshold_ratio=0.15):
            if len(profile) == 0: return 0.0
            peak_val = profile[0] if profile[0] > 0 else np.max(profile[:5])
            threshold = peak_val * threshold_ratio
            energy = 0.0
            for val in profile:
                if val < threshold: break
                energy += float(val)
            return energy

        e_up = ray_energy(prof_up)
        e_down = ray_energy(prof_down)
        e_left = ray_energy(prof_left)
        e_right = ray_energy(prof_right)

        def symmetry_score(e1, e2):
            if e1 + e2 < 1e-6: return 0.0
            return 1.0 - abs(e1 - e2) / (e1 + e2)

        sym_vert = symmetry_score(e_up, e_down)
        sym_horiz = symmetry_score(e_left, e_right)

        both_exist_vert = 1.0 if (e_up > 0 and e_down > 0) else 0.0
        both_exist_horiz = 1.0 if (e_left > 0 and e_right > 0) else 0.0

        total_score = (sym_vert + sym_horiz) * both_exist_vert * both_exist_horiz

        if total_score > best_score:
            best_score = total_score
            best_peak = peak

    if best_peak is None:
        best_peak = max(peaks, key=lambda p: p['strength'])

    return float(best_peak['x']), float(best_peak['y']), best_score

def refine_center_by_projections(gray_img, cx, cy, window_size=12):

    h_img, w_img = gray_img.shape[:2]
    
    x_start = max(0, int(round(cx)) - window_size)
    y_start = max(0, int(round(cy)) - window_size)
    x_end = min(w_img, int(round(cx)) + window_size + 1)
    y_end = min(h_img, int(round(cy)) + window_size + 1)
    
    roi = gray_img[y_start:y_end, x_start:x_end]
    if roi.shape[0] < 5 or roi.shape[1] < 5:
        return cx, cy

    bg_level = np.median(roi)
    roi_clean = np.clip(roi.astype(np.float32) - bg_level, 0, 255)

    if np.max(roi_clean) < 10: 
        return cx, cy 

    proj_x = np.sum(roi_clean, axis=0) 
    proj_y = np.sum(roi_clean, axis=1)

    peak_x_idx = np.argmax(proj_x)
    peak_y_idx = np.argmax(proj_y)

    search_radius = 4

    def get_subpixel_center(proj, peak_idx):
        start = max(0, peak_idx - search_radius)
        end = min(len(proj), peak_idx + search_radius + 1)
        
        weights = proj[start:end]
        positions = np.arange(start, end)
        
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            return peak_idx
            
        return np.sum(positions * weights) / sum_weights

    local_sub_x = get_subpixel_center(proj_x, peak_x_idx)
    local_sub_y = get_subpixel_center(proj_y, peak_y_idx)

    global_x = x_start + local_sub_x
    global_y = y_start + local_sub_y

    if np.hypot(global_x - cx, global_y - cy) > window_size:
        return cx, cy

    return float(global_x), float(global_y)
def order_points_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_coordinate_system(gray_img, markers):
    marked_crosses = []
    
    for m in markers:
        x, y = int(round(m['x'])), int(round(m['y']))
        R = 45 
        
        y1, y2 = max(0, y-R), min(gray_img.shape[0], y+R)
        x1, x2 = max(0, x-R), min(gray_img.shape[1], x+R)
        roi = gray_img[y1:y2, x1:x2]
        
        if roi.shape[0] < R or roi.shape[1] < R: continue
        
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        
        edges = cv2.Canny(blur, 20, 80)
        
        kernel = np.ones((4, 4), np.uint8)
        edges_closed = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        is_marked = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 800 < area < 8000:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                
                if 0.7 < aspect_ratio < 1.3:
                    dist_to_center = cv2.pointPolygonTest(cnt, (roi.shape[1]//2, roi.shape[0]//2), False)
                    if dist_to_center >= 0:
                        is_marked = True
                        break
                        
        if is_marked:
            marked_crosses.append(m)

    if len(marked_crosses) < 3:
        print(f"=> ВНИМАНИЕ: Найдено кружков: {len(marked_crosses)}. Оси не установлены.")
        return None

    pts = np.array([[m['x'], m['y']] for m in marked_crosses])
    dists_nn = []
    for p in pts:
        ds = np.linalg.norm(pts - p, axis=1)
        ds = ds[ds > 0] 
        if len(ds) > 0: dists_nn.append(np.min(ds))
        
    if not dists_nn: return None
    step_px = np.median(dists_nn)
    
    grouping_threshold = step_px * 3.5 
    
    groups = []
    for m in marked_crosses:
        placed = False
        for g in groups:
            if math.hypot(m['x'] - g[0]['x'], m['y'] - g[0]['y']) < grouping_threshold:
                g.append(m)
                placed = True
                break
        if not placed:
            groups.append([m])

    axes_data_list = []
    
    for i, group in enumerate(groups):
        if len(group) >= 3: 
            m1, m2, m3 = group[:3] 
            
            def get_dist(a, b): return math.hypot(a['x'] - b['x'], a['y'] - b['y'])
            
            distances = {
                (0, 1): get_dist(m1, m2), 
                (1, 2): get_dist(m2, m3), 
                (0, 2): get_dist(m1, m3)
            }
            
            closest_pair = min(distances, key=distances.get)
            
            if closest_pair == (0, 1): cand_orig1, cand_orig2, cand_y = m1, m2, m3
            elif closest_pair == (1, 2): cand_orig1, cand_orig2, cand_y = m2, m3, m1
            else: cand_orig1, cand_orig2, cand_y = m1, m3, m2
            
            if get_dist(cand_orig1, cand_y) < get_dist(cand_orig2, cand_y): 
                origin, axis_x = cand_orig1, cand_orig2
            else: 
                origin, axis_x = cand_orig2, cand_orig1
                
            axes_data_list.append({
                'origin': origin, 
                'x': axis_x, 
                'y': cand_y,
                'group_id': i + 1
            })

    if len(axes_data_list) > 0:
        print(f"Установлено систем координат: {len(axes_data_list)}.")
        return axes_data_list
    else:
        print("Кружки найдены, но не образуют полные системы.")
        return None