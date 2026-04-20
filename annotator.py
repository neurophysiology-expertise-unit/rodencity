import cv2
import numpy as np
import pandas as pd
import os
import json
import multiprocessing
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QSlider, QSpinBox, QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap

# ----- Parallel Worker Logic ------
def process_video_chunk_auto_mask(args):
    video_path, start_f, end_f, baseline_bgr, spin_thresh_value, invert_checked, arena_history, mask_folder = args
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    results = []
    
    for i in range(start_f, end_f):
        ret, frame_img = cap.read()
        if not ret: break
        
        diff = cv2.absdiff(frame_img, baseline_bgr)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, spin_thresh_value, 255, cv2.THRESH_BINARY)
        
        if invert_checked:
            thresh = cv2.bitwise_not(thresh)
            
        sorted_frames = sorted([int(k) for k in arena_history.keys()])
        active = None
        for f in sorted_frames:
            if f <= i: active = arena_history[f]
            else: break
            
        if active:
            if len(active) == 4 and isinstance(active[0], (int, float)):
                x1, y1, x2, y2 = active
                active = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            p = np.array(active, dtype=np.int32).reshape((-1, 1, 2))
            mask = np.zeros_like(thresh)
            cv2.fillPoly(mask, [p], 255)
            thresh = cv2.bitwise_and(thresh, mask)
            
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        import os
        cv2.imwrite(os.path.join(mask_folder, f"mask_{i:04d}.png"), closed)
        
        is_painted = (closed > 0)
        mean_val = np.mean(is_painted)
        std_val = np.std(is_painted)
        results.append({'Frame': i, 'Mean_Density': float(mean_val), 'Std_Density': float(std_val)})
        
    cap.release()
    return results

# ----- Main Application ------
class VideoAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mouse Video Annotator (Parallel Auto-Mask)")
        
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        
        self.frame_bgr = None
        self.mask = None
        self.baseline_bgr = None
        
        self.brush_size = 10
        self.is_erasing = False
        self.is_drawing = False
        
        self.defining_arena = False
        self.arena_polygon_pts = []
        self.arena_history = {}
        
        self.mask_folder = None
        self.stats_file = None
        self.arena_file = None
        
        self.setFocusPolicy(Qt.StrongFocus)
        self.init_ui()
        
    def init_ui(self):
        v_layout = QVBoxLayout()
        
        # --- Top Navigation ---
        h_nav = QHBoxLayout()
        self.btn_load = QPushButton("1. Load Video")
        self.btn_load.clicked.connect(self.load_video)
        
        self.btn_prev = QPushButton("< Prev (A)")
        self.btn_prev.clicked.connect(self.prev_frame)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.slider_moved)
        self.btn_next = QPushButton("Next (D) >")
        self.btn_next.clicked.connect(self.next_frame)
        self.lbl_frame_info = QLabel("Frame: 0/0")
        
        h_nav.addWidget(self.btn_load)
        h_nav.addWidget(self.btn_prev)
        h_nav.addWidget(self.slider)
        h_nav.addWidget(self.btn_next)
        h_nav.addWidget(self.lbl_frame_info)
        
        # --- Auto Subtraction Controls ---
        h_auto = QHBoxLayout()
        self.btn_calc_baseline = QPushButton("2. Calc Baseline")
        self.btn_calc_baseline.clicked.connect(self.calc_baseline)
        
        self.spin_thresh = QSpinBox()
        self.spin_thresh.setRange(1, 255)
        self.spin_thresh.setValue(30)
        self.spin_thresh.setToolTip("Lower = picks up faint things. Higher = blocks shadow.")
        
        self.chk_invert = QCheckBox("Invert Detection")
        self.chk_invert.setToolTip("If the floor becomes detected instead of the mice, check this.")
        
        self.btn_auto_current = QPushButton("Auto Mask Current")
        self.btn_auto_current.clicked.connect(self.apply_auto_mask_current)
        self.btn_auto_all = QPushButton("Auto Mask ALL (Parallel)")
        self.btn_auto_all.setStyleSheet("background-color: #6daee2;")
        self.btn_auto_all.clicked.connect(self.apply_auto_mask_all)
        
        h_auto.addWidget(self.btn_calc_baseline)
        h_auto.addWidget(QLabel("Thresh:"))
        h_auto.addWidget(self.spin_thresh)
        h_auto.addWidget(self.chk_invert)
        h_auto.addWidget(self.btn_auto_current)
        h_auto.addWidget(self.btn_auto_all)

        # --- Manual Controls ---
        h_ctrl = QHBoxLayout()
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 100)
        self.spin_brush.setValue(self.brush_size)
        self.spin_brush.valueChanged.connect(self.change_brush)
        
        self.chk_erase = QCheckBox("Erase Mode")
        self.chk_erase.stateChanged.connect(self.toggle_erase)
        
        self.btn_interpolate = QPushButton("Interpolate Gap")
        self.btn_interpolate.clicked.connect(self.interpolate_gap)
        
        self.btn_define_arena = QPushButton("3. Define 4-Point Arena")
        self.btn_define_arena.setStyleSheet("background-color: #ffd966;")
        self.btn_define_arena.clicked.connect(self.start_arena_definition)
        
        self.btn_export = QPushButton("4. Export Final Video")
        self.btn_export.setStyleSheet("background-color: #93c47d;")
        self.btn_export.clicked.connect(self.export_video)
        
        h_ctrl.addWidget(QLabel("Brush Size:"))
        h_ctrl.addWidget(self.spin_brush)
        h_ctrl.addWidget(self.chk_erase)
        h_ctrl.addWidget(self.btn_interpolate)
        h_ctrl.addWidget(self.btn_define_arena)
        h_ctrl.addWidget(self.btn_export)
        
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMouseTracking(True)
        self.lbl_image.mousePressEvent = self.mouse_press
        self.lbl_image.mouseMoveEvent = self.mouse_move
        self.lbl_image.mouseReleaseEvent = self.mouse_release
        
        v_layout.addWidget(self.lbl_image)
        v_layout.addLayout(h_nav)
        v_layout.addLayout(h_auto)
        v_layout.addLayout(h_ctrl)
        
        self.setLayout(v_layout)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.prev_frame()
        elif event.key() == Qt.Key_D:
            self.next_frame()
        elif event.key() == Qt.Key_E:
            self.chk_erase.setChecked(True)
        elif event.key() == Qt.Key_W:
            self.chk_erase.setChecked(False)
        else:
            super().keyPressEvent(event)
            
    def change_brush(self, val): self.brush_size = val
    def toggle_erase(self, state): self.is_erasing = state == Qt.Checked
        
    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.avi *.mp4)")
        if not path: return
            
        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames == 0: return
            
        self.current_frame_idx = 0
        self.baseline_bgr = None
        self.slider.setEnabled(True)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.setValue(0)
        
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.mask_folder = os.path.join(os.path.dirname(self.video_path), f"{video_name}_masks")
        if not os.path.exists(self.mask_folder):
            os.makedirs(self.mask_folder)
            
        self.stats_file = os.path.join(self.mask_folder, "density_stats.csv")
        self.arena_file = os.path.join(self.mask_folder, "arena_bounds.json")
        
        if os.path.exists(self.arena_file):
            with open(self.arena_file, "r") as f:
                data = json.load(f)
                self.arena_history = {int(k): v for k, v in data.items()}
        else:
            self.arena_history = {}
            
        self.read_frame()
        self.update_display()
        
    # -------- EXPORT VIDEO --------
    def export_video(self):
        if self.cap is None: return
        export_path = os.path.splitext(self.video_path)[0] + "_labeled.avi"
        
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(export_path, fourcc, fps, (w, h))
        
        self.btn_export.setText("...Exporting...")
        QApplication.processEvents()
        
        for i in range(self.total_frames):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if not ret: break
            
            m_path = os.path.join(self.mask_folder, f"mask_{i:04d}.png")
            if os.path.exists(m_path):
                mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    overlay = frame.copy()
                    overlay[mask > 0] = [0, 0, 255]
                    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
            poly = self.get_active_arena_poly(i)
            if poly:
                p = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [p], True, (0, 255, 0), 2)
            
            out.write(frame)
            if i % 10 == 0:
                self.lbl_frame_info.setText(f"Exporting: {i}/{self.total_frames}")
                QApplication.processEvents()
                
        out.release()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.read_frame()
        self.update_display()
        self.btn_export.setText("4. Export Final Video")
        QMessageBox.information(self, "Export Complete", f"Successfully exported labeled video to:\n{export_path}")
        
    # -------- Arena Definition --------
    def start_arena_definition(self):
        self.defining_arena = True
        self.arena_polygon_pts = []
        QMessageBox.information(self, "Define Arena", "Click the 4 corners of your arena in order (e.g. top-left, top-right, bottom-right, bottom-left).")
        
    def get_active_arena_poly(self, frame_idx):
        if not self.arena_history: return None
        sorted_frames = sorted(list(self.arena_history.keys()))
        active = None
        for f in sorted_frames:
            if f <= frame_idx: active = self.arena_history[f]
            else: break
            
        if active and len(active) == 4 and isinstance(active[0], (int, float)):
            x1, y1, x2, y2 = active
            active = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
        return active
        
    def save_arena_poly(self, poly, frame_idx):
        self.arena_history[frame_idx] = poly
        with open(self.arena_file, "w") as f:
            json.dump(self.arena_history, f)
        self.update_display()
        
    # -------- Auto Background Subtraction --------
    def calc_baseline(self):
        if self.cap is None: return
        self.btn_calc_baseline.setText("...Calculating...")
        QApplication.processEvents()
        
        frames = []
        step = max(1, self.total_frames // 30)
        for i in range(0, self.total_frames, step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, fr = self.cap.read()
            if ret: frames.append(fr)
                
        if len(frames) > 0:
            self.baseline_bgr = np.median(frames, axis=0).astype(np.uint8)
            QMessageBox.information(self, "Success", "Baseline median computed successfully!")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.btn_calc_baseline.setText("2. Calc Baseline")
        
    def auto_mask_frame(self, frame_img, frame_idx):
        if self.baseline_bgr is None: return None
        
        diff = cv2.absdiff(frame_img, self.baseline_bgr)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, self.spin_thresh.value(), 255, cv2.THRESH_BINARY)
        
        if self.chk_invert.isChecked():
            thresh = cv2.bitwise_not(thresh)
        
        poly = self.get_active_arena_poly(frame_idx)
        if poly:
            p = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            mask = np.zeros_like(thresh)
            cv2.fillPoly(mask, [p], 255)
            thresh = cv2.bitwise_and(thresh, mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return closed
        
    def apply_auto_mask_current(self):
        if self.baseline_bgr is None:
            QMessageBox.warning(self, "Warning", "Please calculate baseline first!")
            return
        m = self.auto_mask_frame(self.frame_bgr, self.current_frame_idx)
        if m is not None:
            self.mask = m
            self.save_current_mask()
            self.update_display()
            
    def apply_auto_mask_all(self):
        if self.baseline_bgr is None:
            QMessageBox.warning(self, "Warning", "Please calculate baseline first!")
            return
            
        reply = QMessageBox.question(self, "Confirm", "Process all frames in Parallel? Existing masks will be overwritten.", QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes: return
        
        self.btn_auto_all.setText("...Processing in Parallel...")
        QApplication.processEvents()
        
        n_cpus = multiprocessing.cpu_count()
        chunk_size = max(1, self.total_frames // (n_cpus * 2))
        
        chunks = []
        for i in range(0, self.total_frames, chunk_size):
            chunks.append((self.video_path, i, min(self.total_frames, i + chunk_size), 
                           self.baseline_bgr, self.spin_thresh.value(), 
                           self.chk_invert.isChecked(), self.arena_history, self.mask_folder))
                           
        with multiprocessing.Pool(n_cpus) as pool:
            results = pool.map(process_video_chunk_auto_mask, chunks)
            
        # Combine all partial CSV chunks and rewrite the CSV safely
        flat_results = [item for sublist in results for item in sublist]
        pd.DataFrame(flat_results).sort_values(by='Frame').to_csv(self.stats_file, index=False)
        
        self.btn_auto_all.setText("Auto Mask ALL (Parallel)")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.read_frame()
        self.update_display()
        QMessageBox.information(self, "Done", f"All {self.total_frames} frames processed instantly across {n_cpus} CPUs!")
        
    # -------- Navigation & Core --------
    def prev_frame(self): self.slider.setValue(max(0, self.current_frame_idx - 1))
    def next_frame(self): self.slider.setValue(min(self.total_frames - 1, self.current_frame_idx + 1))
        
    def slider_moved(self, val):
        self.save_current_mask()
        self.current_frame_idx = val
        self.read_frame()
        self.update_display()
        
    def read_frame(self):
        if self.cap is None: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        was_superimposed = False
        if ret:
            self.frame_bgr = frame
            mask_path = os.path.join(self.mask_folder, f"mask_{self.current_frame_idx:04d}.png")
            if os.path.exists(mask_path):
                self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                if self.mask is None or self.mask.shape[:2] != self.frame_bgr.shape[:2]:
                    self.mask = np.zeros(self.frame_bgr.shape[:2], dtype=np.uint8)
                else:
                    self.mask = self.mask.copy()
                    was_superimposed = True
        else:
            self.frame_bgr = None
            
        self.lbl_frame_info.setText(f"Frame: {self.current_frame_idx}/{self.total_frames - 1}")
        if was_superimposed: self.save_current_mask()
        
    # -------- Utils & IO --------
    def clear_mask(self):
        if self.mask is not None:
            self.mask.fill(0)
            self.update_display()
            self.save_current_mask()
            
    def save_current_mask(self):
        if self.mask is None: return
        mask_path = os.path.join(self.mask_folder, f"mask_{self.current_frame_idx:04d}.png")
        cv2.imwrite(mask_path, self.mask)
        self.update_stats(self.current_frame_idx, self.mask)
        
    def interpolate_gap(self):
        if self.current_frame_idx == 0 or self.mask is None: return
        prev_idx = -1
        for i in range(self.current_frame_idx - 1, -1, -1):
            if os.path.exists(os.path.join(self.mask_folder, f"mask_{i:04d}.png")):
                prev_idx = i; break
        if prev_idx == -1: return
            
        mask_prev = cv2.imread(os.path.join(self.mask_folder, f"mask_{prev_idx:04d}.png"), cv2.IMREAD_GRAYSCALE)
        mask_curr = self.mask
        
        M_p = cv2.moments(mask_prev)
        M_c = cv2.moments(mask_curr)
        cx_p = M_p['m10']/M_p['m00'] if M_p['m00']>0 else 0
        cy_p = M_p['m01']/M_p['m00'] if M_p['m00']>0 else 0
        cx_c = M_c['m10']/M_c['m00'] if M_c['m00']>0 else 0
        cy_c = M_c['m01']/M_c['m00'] if M_c['m00']>0 else 0
        
        total_steps = self.current_frame_idx - prev_idx
        for step in range(1, total_steps):
            idx = prev_idx + step
            w = step / total_steps
            M_trans = np.float32([[1, 0, int((cx_c - cx_p)*w)], [0, 1, int((cy_c - cy_p)*w)]])
            shifted = cv2.warpAffine(mask_prev, M_trans, (mask_prev.shape[1], mask_prev.shape[0]))
            cv2.imwrite(os.path.join(self.mask_folder, f"mask_{idx:04d}.png"), shifted)
            self.update_stats(idx, shifted)
        QMessageBox.information(self, "Success", f"Interpolated {total_steps - 1} frames.")
        
    def update_stats(self, frame_idx, mask_array):
        is_painted = (mask_array > 0)
        mean_val = np.mean(is_painted)
        std_val = np.std(is_painted)
        
        data = []
        if os.path.exists(self.stats_file):
            data = pd.read_csv(self.stats_file).to_dict('records')
            
        found = False
        for row in data:
            if row['Frame'] == frame_idx:
                row['Mean_Density'], row['Std_Density'] = mean_val, std_val
                found = True; break
        if not found:
            data.append({'Frame': frame_idx, 'Mean_Density': mean_val, 'Std_Density': std_val})
            
        pd.DataFrame(data).sort_values(by='Frame').to_csv(self.stats_file, index=False)
        
    def update_display(self):
        if self.frame_bgr is None: return
        overlay = self.frame_bgr.copy()
        
        poly = self.get_active_arena_poly(self.current_frame_idx)
        if poly is not None:
            p = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [p], isClosed=True, color=(0, 255, 0), thickness=2)
            
        if self.defining_arena and len(self.arena_polygon_pts) > 0:
            pts = self.arena_polygon_pts
            for i in range(len(pts)):
                cv2.circle(overlay, (pts[i][0], pts[i][1]), 4, (0, 255, 255), -1)
                if i > 0:
                    cv2.line(overlay, (pts[i-1][0], pts[i-1][1]), (pts[i][0], pts[i][1]), (0, 255, 255), 2)
            if len(pts) == 4:
                cv2.line(overlay, (pts[-1][0], pts[-1][1]), (pts[0][0], pts[0][1]), (0, 255, 255), 2)
            
        overlay[self.mask > 0] = [0, 0, 255]
        blended = cv2.addWeighted(overlay, 0.4, self.frame_bgr, 0.6, 0)
        
        rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.lbl_image.setPixmap(QPixmap.fromImage(qimg))
        
    # -------- Mouse Interactions --------
    def draw_on_mask(self, pos):
        if self.mask is None or self.frame_bgr is None: return
        cv2.circle(self.mask, (pos.x(), pos.y()), self.brush_size, 0 if self.is_erasing else 255, -1)
        self.update_display()
        
    def mouse_press(self, event):
        if self.defining_arena: pass
        elif event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.draw_on_mask(event.pos())
            
    def mouse_move(self, event):
        if self.is_drawing and not self.defining_arena:
            self.draw_on_mask(event.pos())
            
    def mouse_release(self, event):
        if self.defining_arena:
            if event.button() == Qt.LeftButton:
                x, y = event.pos().x(), event.pos().y()
                self.arena_polygon_pts.append([x, y])
                
                if len(self.arena_polygon_pts) == 4:
                    self.save_arena_poly(self.arena_polygon_pts, self.current_frame_idx)
                    QMessageBox.information(self, "Saved", f"Tilted arena boundaries saved for frame {self.current_frame_idx} onwards!")
                    self.defining_arena = False
                    self.arena_polygon_pts = []
                    
                self.update_display()
        else:
            if event.button() == Qt.LeftButton:
                self.is_drawing = False
                self.save_current_mask()
