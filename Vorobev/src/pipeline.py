import cv2
import numpy as np
import json
import os
import math

from .corrector import PerspectiveCorrector
from .detector import detect_coordinate_system
from .detector import process_with_params
from .detector import refine_center_by_projections
from .filters import (calculate_penalty, filter_by_local_contrast)
from .visualizer import draw_step_markers, draw_final_results, plot_process
from .detector import refine_center_by_projections, find_cross_rough_center
from .visualizer import plot_single_cross_process 

def find_cross_markers_auto(image_path, grid_shape=(7, 7), step_mm=35, use_aruco=False, find_axes=False,
                            output_json="markers.json", output_img="result.jpg", output_plot_path="plot_photo.jpg",
                            show_process=True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")
    
    image = cv2.imread(image_path)
    original_vis = image.copy()
    gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print(f"\n[{os.path.basename(image_path)}] Запуск...")

    sheet_mask = None
    if sheet_mask is None:
        blur = cv2.GaussianBlur(gray_orig, (5, 5), 0)
        _, sheet_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        sheet_contours, _ = cv2.findContours(sheet_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(sheet_contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = np.intp(cv2.boxPoints(rect))
        sheet_mask = np.zeros_like(gray_orig)
        cv2.drawContours(sheet_mask, [box], 0, 255, thickness=cv2.FILLED)
        margin_kernel = np.ones((45, 45), np.uint8)
        sheet_mask = cv2.erode(sheet_mask, margin_kernel, iterations=1)

    profiles = [
        {"name": "Без стекла", "alpha": 2.0, "tophat": 25, "thresh": 40},
        {"name": "Стекло + Черный потолок",  "alpha": 3.0, "tophat": 35, "thresh": 40}, 
        {"name": "Стекло + Блики",           "alpha": 1.1, "tophat": 15, "thresh": 20}
    ]

    # Переменные для хранения победившего профиля
    best_markers_initial = []
    best_filtered_1 = []
    best_filtered_2 = []
    # best_filtered_3 = []
    # best_filtered_4 = []
    
    best_profile = None
    best_score = float('inf')
    best_images = {}
    expected_count = grid_shape[0] * grid_shape[1]

    for params in profiles:
        raw_markers, fin_thresh, bright_img = process_with_params(image, sheet_mask, params)
        
        # Прогон 1 (Площадь)
        current_filtered_1 = []
        if len(raw_markers) > 0:
            median_area = np.median([m['area'] for m in raw_markers])
            current_filtered_1 = [m for m in raw_markers if (median_area * 0.3) < m['area'] < (median_area * 2)]
            
        # Прогон 2 (Локальный контраст)
        current_filtered_2 = filter_by_local_contrast(current_filtered_1, gray_orig)
        
        # Прогон 3 (Форма Хафа)
        # current_filtered_3 = filter_by_hough_cross(current_filtered_2, fin_thresh)
        
        # Прогон 4 (NMS)
        # current_filtered_4 = filter_by_grid_distance(current_filtered_3)

        score = calculate_penalty(current_filtered_2, expected_count)
        
        if score < best_score:
            best_score = score
            best_profile = params
            best_markers_initial = raw_markers
            best_filtered_1 = current_filtered_1
            best_filtered_2 = current_filtered_2
            # best_filtered_3 = current_filtered_3
            # best_filtered_4 = current_filtered_4
            best_images = {"thresh": fin_thresh, "bright": bright_img}

    filtered_1 = best_filtered_1
    filtered_2 = best_filtered_2
    # filtered_3 = best_filtered_3
    # filtered_4 = best_filtered_4

    vis_step0 = draw_step_markers(image, best_markers_initial)
    vis_step1 = draw_step_markers(image, filtered_1)
    vis_step2 = draw_step_markers(image, filtered_2)
    # vis_step3 = draw_step_markers(image, filtered_3)
    # vis_step4 = draw_step_markers(image, filtered_4)

    if len(filtered_2) > 0:
        points = np.array([[[m['x'], m['y']]] for m in filtered_2], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        points_subpix = cv2.cornerSubPix(gray_orig, points, (5, 5), (-1, -1), criteria)
        for i, m in enumerate(filtered_2):
            m['x'] = float(points_subpix[i][0][0])
            m['y'] = float(points_subpix[i][0][1])
            m['type'] = 'point'

    # tolerance = expected_count + (expected_count * 0.2)
    
    # if len(filtered_2) <= tolerance:
    # if False:
    pts_for_corrector = [(m['x'], m['y']) for m in filtered_2]
    # else:
        # pts_for_corrector = [(m['x'], m['y']) for m in filtered_4]

    corrector = PerspectiveCorrector(grid_size=grid_shape, actual_step_mm=step_mm)
    restored_pixels = corrector.process_and_restore_grid(pts_for_corrector)
    
    final_markers = []
    if len(restored_pixels) == expected_count:
        for rx, ry in restored_pixels:
            
            clean_cands = [m for m in filtered_2 if math.hypot(m['x'] - rx, m['y'] - ry) < 10.0]
            if clean_cands:
                matched_m = min(clean_cands, key=lambda m: math.hypot(m['x'] - rx, m['y'] - ry))
                matched_m['is_virtual'] = False
                matched_m['type'] = 'point'
                final_markers.append(matched_m)
                continue
                
            dirty_cands = [m for m in best_markers_initial if math.hypot(m['x'] - rx, m['y'] - ry) < 10.0]
            if dirty_cands:
                matched_m = min(dirty_cands, key=lambda m: math.hypot(m['x'] - rx, m['y'] - ry))
                matched_m['is_virtual'] = False
                matched_m['type'] = 'point'
                final_markers.append(matched_m)
                continue

            final_markers.append({'x': rx, 'y': ry, 'box': (int(rx)-5, int(ry)-5, 10, 10), 'is_virtual': True, 'type': 'point'})
    else:
        final_markers = filtered_2


    real_final_markers = [m for m in final_markers if not m.get('is_virtual', False)]
    if len(real_final_markers) > 0:
        for m in real_final_markers:
            new_x, new_y = refine_center_by_projections(gray_orig, m['x'], m['y'], window_size=12)
            m['x'] = new_x
            m['y'] = new_y
    print(f"=> Победил профиль: '{best_profile['name']}'")
    print(f"   Найдено кандидатов: {len(best_markers_initial)}")
    print(f"   После прогона 1 (площадь): {len(filtered_1)}")
    print(f"   После прогона 2 (контраст): {len(filtered_2)}")
    # print(f"   После прогона 3 (Хаф крест): {len(filtered_3)}")
    print(f"   Итог после всех фильтров: {len(final_markers)}")

    axes_data_list = None
    if find_axes and len(final_markers) > 0:
        axes_data_list = detect_coordinate_system(gray_orig, final_markers)
        if axes_data_list:
            for axes in axes_data_list:
                gid = axes['group_id']
                for m in final_markers:
                    if m is axes['origin']: m['type'] = f'origin_{gid}'
                    elif m is axes['x']: m['type'] = f'axis_x_{gid}'
                    elif m is axes['y']: m['type'] = f'axis_y_{gid}'


    draw_final_results(original_vis, final_markers, axes_data_list)
    cv2.imwrite(output_img, original_vis)

    json_data = [{
        "x": round(m["x"], 2),
        "y": round(m["y"], 2),
        "is_virtual": m.get("is_virtual", False),
        "type": m.get("type", "point")
    } for m in final_markers]
        
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    if show_process or output_plot_path:
        # steps_images = [vis_step0, vis_step1, vis_step2, vis_step3, vis_step4]
        steps_images = [vis_step0, vis_step1, vis_step2]
        # steps_counts = [len(best_markers_initial), len(filtered_1), len(filtered_2), len(filtered_3), len(filtered_4)]
        steps_counts = [len(best_markers_initial), len(filtered_1), len(filtered_2)]
        
        plot_process(
            image, best_images["thresh"], steps_images, steps_counts, 
            original_vis, final_markers, 
            output_plot_path=output_plot_path, 
            show_process=show_process 
        )
        
    return json_data, original_vis

def find_single_cross(image_path, output_json="single_marker.json",
                      output_img="single_result.jpg",
                      output_plot_path=None, show_process=True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    original_vis = image.copy()
    gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f"\n[{os.path.basename(image_path)}] Запуск ОДИНОЧНОГО режима...")

    rough_cx, rough_cy, box, best_thresh, steps = find_cross_rough_center(gray_orig)

    if rough_cx is None:
        print(f"   [ВНИМАНИЕ] Крестик не обнаружен на изображении!")

        json_data = [{
            "x": None,
            "y": None,
            "found": False,
            "type": "single_cross"
        }]
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

        warn_vis = image.copy()
        cv2.putText(warn_vis, "CROSS NOT FOUND", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imwrite(output_img, warn_vis)

        if show_process or output_plot_path:
            steps.insert(0, (image.copy(), "0. Исходник"))
            plot_single_cross_process(
                image, steps, 0, 0,
                output_plot_path=output_plot_path,
                show_process=show_process
            )

        return json_data, warn_vis

    if box is not None:
        win_size = int(max(box[2], box[3]) * 0.5)
        win_size = int(np.clip(win_size, 15, 80))
    else:
        win_size = 25

    final_x, final_y = refine_center_by_projections(
        gray_orig, rough_cx, rough_cy, window_size=win_size
    )
    print(f"   Результат: X={final_x:.3f}, Y={final_y:.3f}")

    ix, iy = int(round(final_x)), int(round(final_y))
    if box:
        bx, by, bw, bh = box
        cv2.rectangle(original_vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
    cv2.circle(original_vis, (ix, iy), 1, (0, 0, 255), -1)
    cv2.drawMarker(original_vis, (ix, iy), (0, 0, 255),
                   markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=1)
    cv2.imwrite(output_img, original_vis)

    if show_process or output_plot_path:
        steps.insert(0, (image.copy(), "0. Исходник"))
        roi_vis = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR)
        R = win_size
        cv2.rectangle(roi_vis,
                      (int(rough_cx) - R, int(rough_cy) - R),
                      (int(rough_cx) + R, int(rough_cy) + R),
                      (255, 0, 0), 2)
        cv2.circle(roi_vis, (int(rough_cx), int(rough_cy)), 4, (0, 255, 0), -1)
        steps.append((roi_vis, f"4. ROI для проекций (win={win_size})"))
        plot_single_cross_process(
            image, steps, final_x, final_y,
            output_plot_path=output_plot_path,
            show_process=show_process
        )

    json_data = [{
        "x": round(final_x, 3),
        "y": round(final_y, 3),
        "found": True,
        "type": "single_cross"
    }]
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    return json_data, original_vis