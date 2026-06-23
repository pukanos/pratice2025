import cv2
import matplotlib.pyplot as plt
import math

def draw_step_markers(image, markers):
    vis = image.copy()
    for m in markers:
        cx, cy = int(m["x"]), int(m["y"])
        x, y, w, h = m["box"]
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
    return vis

def draw_final_results(original_vis, final_markers, axes_data_list=None):
    for m in final_markers:
        cx, cy = int(m["x"]), int(m["y"])
        if m.get('is_virtual', False):
            cv2.drawMarker(original_vis, (cx, cy), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
            cv2.circle(original_vis, (cx, cy), 6, (255, 255, 0), 1)
        else:
            x, y, w, h = m["box"]
            cv2.rectangle(original_vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.circle(original_vis, (cx, cy), 1, (0, 0, 255), -1)

    if axes_data_list:
        for axes in axes_data_list:
            ox, oy = int(round(axes['origin']['x'])), int(round(axes['origin']['y']))
            xx, xy = int(round(axes['x']['x'])), int(round(axes['x']['y']))
            yx, yy = int(round(axes['y']['x'])), int(round(axes['y']['y']))
            
            cv2.circle(original_vis, (ox, oy), 15, (0, 255, 255), 2)
            
            cv2.arrowedLine(original_vis, (ox, oy), (xx, xy), (0, 0, 255), 3, tipLength=0.15)
            cv2.arrowedLine(original_vis, (ox, oy), (yx, yy), (0, 255, 0), 3, tipLength=0.15)
            
            gid = axes['group_id']
            cv2.putText(original_vis, f"X{gid}", (xx+5, xy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(original_vis, f"Y{gid}", (yx+5, yy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def plot_process(image, thresh_img, steps_images, steps_counts, original_vis, final_markers, output_plot_path=None, show_process=True):
    # Уменьшим ширину окна, так как колонок теперь 3, а не 4
    plt.figure(figsize=(15, 10)) 
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("1. Исходник")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(thresh_img, cmap='gray')
    plt.title(f"2. Маска (Кандидатов: {steps_counts[0]})")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(steps_images[0], cv2.COLOR_BGR2RGB))
    plt.title(f"3. Сырые метки: {steps_counts[0]}")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(steps_images[1], cv2.COLOR_BGR2RGB))
    plt.title(f"4. По площади: {steps_counts[1]}")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(steps_images[2], cv2.COLOR_BGR2RGB))
    plt.title(f"5. По контрасту: {steps_counts[2]}")
    plt.axis('off')

    virtual_count = sum(1 for m in final_markers if m.get('is_virtual'))
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(original_vis, cv2.COLOR_BGR2RGB))
    plt.title(f"6. ИТОГ: {len(final_markers)} (Виртуальных: {virtual_count})")
    plt.axis('off')

    plt.tight_layout()
    
    if output_plot_path:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')

    if show_process:
        plt.show()
        
    plt.close()

def plot_single_cross_process(image, steps, final_x, final_y, output_plot_path=None, show_process=True):
    n = len(steps) + 1  
    cols = 3
    rows = math.ceil(n / cols)
    
    plt.figure(figsize=(cols * 5, rows * 4))
    
    for i, (img, title) in enumerate(steps):
        plt.subplot(rows, cols, i + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    final_vis = image.copy()
    ix, iy = int(round(final_x)), int(round(final_y))
    cv2.circle(final_vis, (ix, iy), 1, (0, 0, 255), -1)
    cv2.drawMarker(final_vis, (ix, iy), (0, 0, 255), 
                   markerType=cv2.MARKER_TILTED_CROSS, markerSize=15, thickness=2)
    
    plt.subplot(rows, cols, len(steps) + 1)
    plt.imshow(cv2.cvtColor(final_vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Итог: X={final_x:.2f}, Y={final_y:.2f}")
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_plot_path:
        plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    if show_process:
        plt.show()
    plt.close()

def draw_single_cross_result(original_vis, final_x, final_y, box):
    ix, iy = int(round(final_x)), int(round(final_y))
    bx, by, bw, bh = box
    
    cv2.rectangle(original_vis, (bx, by), (bx+bw, by+bh), (0, 255, 0), 1)
    cv2.circle(original_vis, (ix, iy), 1, (0, 0, 255), -1)
    cv2.drawMarker(original_vis, (ix, iy), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=1)