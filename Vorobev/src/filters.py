import cv2
import numpy as np
import math

def filter_by_local_contrast(markers, gray_img):
    valid_markers = []
    for m in markers:
        x, y, w, h = m['box']
        roi_inner = gray_img[y:y+h, x:x+w]
        pad_x, pad_y = int(w * 0.8), int(h * 0.8)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(gray_img.shape[1], x + w + pad_x), min(gray_img.shape[0], y + h + pad_y)
        roi_outer = gray_img[y1:y2, x1:x2]
        
        if roi_inner.size == 0 or roi_outer.size == 0:
            continue
            
        max_inner = np.max(roi_inner)
        median_outer = np.median(roi_outer)
        if max_inner > (median_outer + 10): 
            valid_markers.append(m)
    return valid_markers

def filter_by_hough_cross(markers, thresh_img):
    valid_markers = []
    for m in markers:
        x, y, w, h = m['box']
        pad = 2
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(thresh_img.shape[1], x + w + pad), min(thresh_img.shape[0], y + h + pad)
        
        roi = thresh_img[y1:y2, x1:x2]
        if roi.shape[0] < 5 or roi.shape[1] < 5:
            valid_markers.append(m)
            continue
            
        min_line_len = max(3, int(min(w, h) * 0.4))
        lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=4, 
                                minLineLength=min_line_len, maxLineGap=3)
        
        if lines is None:
            continue
            
        has_horiz = False
        has_vert = False
        
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            angle = math.degrees(math.atan2(ly2 - ly1, lx2 - lx1))
            if angle < 0: angle += 180
            
            if angle < 35 or angle > 145:
                has_horiz = True
            elif 55 < angle < 125:
                has_vert = True
                
        if has_horiz and has_vert:
            valid_markers.append(m)
            
    return valid_markers

def filter_by_grid_distance(markers, min_allowed_dist=30):
    if len(markers) < 2:
        return markers
    median_area = np.median([m['area'] for m in markers])
    markers_sorted = sorted(markers, key=lambda m: abs(m['area'] - median_area))
    valid_markers = []
    for m_new in markers_sorted:
        too_close = False
        for m_valid in valid_markers:
            dist = math.hypot(m_new['x'] - m_valid['x'], m_new['y'] - m_valid['y'])
            if dist < min_allowed_dist:
                too_close = True
                break
        if not too_close:
            valid_markers.append(m_new)
    return valid_markers
def filter_single_cross_geometry(contours):
    valid_candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50: continue 
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.0: 
            continue 
            
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        
        solidity = area / hull_area
        if solidity > 0.55: 
            continue

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            valid_candidates.append({
                "cx": cx, "cy": cy, 
                "area": area, 
                "box": (x, y, w, h)
            })
            
    return valid_candidates
def calculate_penalty(markers, expected_count):
    found_count = len(markers)
    if found_count == 0: return float('inf') 
    count_penalty = abs(found_count - expected_count) * 100
    if found_count < 10: return count_penalty + 10000 
    distances = []
    for i, m1 in enumerate(markers):
        min_dist = float('inf')
        for j, m2 in enumerate(markers):
            if i != j:
                dist = math.hypot(m1['x'] - m2['x'], m1['y'] - m2['y'])
                if dist < min_dist: min_dist = dist
        distances.append(min_dist)
    std_dist = np.std(distances)
    geometry_penalty = std_dist * 20  
    areas = [m['area'] for m in markers]
    std_area = np.std(areas)
    size_penalty = std_area * 5
    return count_penalty + geometry_penalty + size_penalty