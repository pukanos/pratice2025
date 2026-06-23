import cv2
import numpy as np

class PerspectiveCorrector:
    def __init__(self, grid_size: tuple = (7, 7), actual_step_mm: float = 35.0):
        self.grid_size = grid_size
        self.actual_step_mm = actual_step_mm

    def _estimate_homography_robust(self, points: np.ndarray) -> tuple:
        ideal_mm_grid = []
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                ideal_mm_grid.append([c * self.actual_step_mm, r * self.actual_step_mm])
        ideal_mm_grid = np.array(ideal_mm_grid, dtype=np.float32)
        if len(points) < 6: return None, ideal_mm_grid, {}
        
        best_h, best_inliers_count, best_mapping = None, -1, {}
        threshold_mm = self.actual_step_mm * 0.45
        np.random.seed(42)
        for _ in range(300):
            base_pt = points[np.random.choice(len(points))]
            closest_indices = np.argsort(np.linalg.norm(points - base_pt, axis=1))[:12]
            if len(closest_indices) < 4: continue
            sample_pts = points[np.random.choice(closest_indices, 4, replace=False)]
            
            center = np.mean(sample_pts, axis=0)
            
            angles = np.arctan2(sample_pts[:, 1] - center[1], sample_pts[:, 0] - center[0])
            
            sort_idx = np.argsort(angles)
            src_quad = sample_pts[sort_idx].astype(np.float32)
            
            for r_offset in range(self.grid_size[0] - 1):
                for c_offset in range(self.grid_size[1] - 1):
                    dst_quad = np.array([
                        [c_offset * self.actual_step_mm, r_offset * self.actual_step_mm],
                        [(c_offset + 1) * self.actual_step_mm, r_offset * self.actual_step_mm],
                        [(c_offset + 1) * self.actual_step_mm, (r_offset + 1) * self.actual_step_mm],
                        [c_offset * self.actual_step_mm, (r_offset + 1) * self.actual_step_mm]
                    ], dtype=np.float32)
                    h_matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)
                    if np.abs(np.linalg.det(h_matrix)) < 1e-6: continue
                    
                    pts_transformed = cv2.perspectiveTransform(np.array([points]), h_matrix)[0]
                    current_inliers, current_mapping = 0, {}
                    for ideal_idx, ideal_pt in enumerate(ideal_mm_grid):
                        dists = np.linalg.norm(pts_transformed - ideal_pt, axis=1)
                        min_idx = np.argmin(dists)
                        if dists[min_idx] <= threshold_mm:
                            current_inliers += 1
                            current_mapping[ideal_idx] = min_idx
                            
                    if current_inliers > best_inliers_count:
                        best_inliers_count, best_h, best_mapping = current_inliers, h_matrix, current_mapping
                        
        if best_h is None or best_inliers_count < 12: return None, ideal_mm_grid, {}
        
        src_refine = np.array([points[real_idx] for ideal_idx, real_idx in best_mapping.items()], dtype=np.float32)
        dst_refine = np.array([ideal_mm_grid[ideal_idx] for ideal_idx, real_idx in best_mapping.items()], dtype=np.float32)
        refined_h, _ = cv2.findHomography(src_refine, dst_refine, 0)

        if refined_h is not None and np.abs(np.linalg.det(refined_h)) > 1e-6:
            final_mapping = {}
            pts_transformed = cv2.perspectiveTransform(np.array([points]), refined_h)[0]
            for ideal_idx, ideal_pt in enumerate(ideal_mm_grid):
                dists = np.linalg.norm(pts_transformed - ideal_pt, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] <= threshold_mm: final_mapping[ideal_idx] = min_idx
            return refined_h, ideal_mm_grid, final_mapping
        return best_h, ideal_mm_grid, best_mapping
    
    def process_and_restore_grid(self, detected_points: list) -> tuple:
        if len(detected_points) < 4: return detected_points, []
        pts = np.array(detected_points, dtype=np.float32)
        h_matrix, ideal_mm_grid, mapping = self._estimate_homography_robust(pts)
        if h_matrix is None: return detected_points, []
        try: h_matrix_inv = np.linalg.inv(h_matrix)
        except np.linalg.LinAlgError: return detected_points, []
        
        final_pixel_points = []
        for ideal_idx, ideal_pt in enumerate(ideal_mm_grid):
            if ideal_idx in mapping:
                real_idx = mapping[ideal_idx]
                final_pixel_points.append(detected_points[real_idx])
            else:
                restored_pixel_pt = cv2.perspectiveTransform(np.array([[[ideal_pt[0], ideal_pt[1]]]], dtype=np.float32), h_matrix_inv)[0][0]
                final_pixel_points.append((float(restored_pixel_pt[0]), float(restored_pixel_pt[1])))
        return final_pixel_points