import cv2
import numpy as np
import pandas as pd
import os
import json
import multiprocessing
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QSlider, QSpinBox, QCheckBox, QMessageBox, QListWidget, QGroupBox
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap

# ----- Parallel Worker Logics ------

def process_video_chunk_auto_mask(args):
    video_path, start_f, end_f, baseline_bgr, spin_thresh_value, invert_checked, arena_history, mask_folder, keep_largest = args
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
        
        if keep_largest:
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                closed.fill(0)
                cv2.drawContours(closed, [largest], -1, 255, -1)
        
        import os
        cv2.imwrite(os.path.join(mask_folder, f"mask_{i:04d}.png"), closed)
        
        is_painted = (closed > 0)
        mean_val = np.mean(is_painted)
        std_val = np.std(is_painted)
        results.append({'Frame': i, 'Mean_Density': float(mean_val), 'Std_Density': float(std_val)})
        
    cap.release()
    return results

def process_video_chunk_clean_artifacts(args):
    start_f, end_f, mask_folder = args
    import cv2
    import numpy as np
    import os
    results = []
    
    for i in range(start_f, end_f):
        m_path = os.path.join(mask_folder, f"mask_{i:04d}.png")
        if os.path.exists(m_path):
            mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                mask.fill(0)
                cv2.drawContours(mask, [largest], -1, 255, -1)
                cv2.imwrite(m_path, mask)
                
            is_painted = (mask > 0)
            mean_val = np.mean(is_painted)
            std_val = np.std(is_painted)
            results.append({'Frame': i, 'Mean_Density': float(mean_val), 'Std_Density': float(std_val)})
            
    return results


# ----- Main Application ------
class VideoAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rodencity Analysis Pipeline")
        
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        
        self.analysis_start_frame = 0
        self.analysis_end_frame = 0
        
        self.frame_bgr = None
        self.mask = None
        self.baseline_bgr = None
        
        self.brush_size = 10
        self.is_erasing = False
        self.is_drawing = False
        
        self.defining_arena = False
        self.arena_polygon_pts = []
        self.arena_history = {}
        
        self.pending_stim_start = None
        self.pending_stim_end = None
        self.stimulus_events = []
        
        self.mask_folder = None
        self.stats_file = None
        self.arena_file = None
        self.stim_file = None
        
        self.setFocusPolicy(Qt.StrongFocus)
        self.init_ui()
        
    def init_ui(self):
        main_h_layout = QHBoxLayout()
        v_layout = QVBoxLayout()
        
        # Step 1: Load & Trim
        gb1 = QGroupBox("Step 1: Video & Time Window")
        l1 = QVBoxLayout()
        
        h1a = QHBoxLayout()
        self.btn_load = QPushButton("Load Video")
        self.btn_load.clicked.connect(self.load_video)
        self.lbl_frame_info = QLabel("Frame: 0/0")
        h1a.addWidget(self.btn_load)
        h1a.addWidget(self.lbl_frame_info)
        h1a.addStretch()
        
        h1b = QHBoxLayout()
        self.btn_set_start = QPushButton("Set Start Time")
        self.btn_set_start.clicked.connect(self.set_start_frame)
        self.btn_set_end = QPushButton("Set End Time")
        self.btn_set_end.clicked.connect(self.set_end_frame)
        self.lbl_window = QLabel("Window: All")
        h1b.addWidget(self.btn_set_start)
        h1b.addWidget(self.btn_set_end)
        h1b.addWidget(self.lbl_window)
        h1b.addStretch()
        
        l1.addLayout(h1a)
        l1.addLayout(h1b)
        gb1.setLayout(l1)
        
        # Step 2: Environment Constraint
        gb2 = QGroupBox("Step 2: Environment Constraint")
        l2 = QHBoxLayout()
        self.btn_calc_baseline = QPushButton("Calc Baseline")
        self.btn_calc_baseline.clicked.connect(self.calc_baseline)
        self.btn_define_arena = QPushButton("Define 4-Point Arena")
        self.btn_define_arena.setStyleSheet("background-color: #ffd966;")
        self.btn_define_arena.clicked.connect(self.start_arena_definition)
        l2.addWidget(self.btn_calc_baseline)
        l2.addWidget(self.btn_define_arena)
        l2.addStretch()
        gb2.setLayout(l2)
        
        # Step 3: Create Masks
        gb3 = QGroupBox("Step 3: Mask Generation")
        l3 = QHBoxLayout()
        self.spin_thresh = QSpinBox()
        self.spin_thresh.setRange(1, 255)
        self.spin_thresh.setValue(30)
        self.chk_invert = QCheckBox("Invert")
        
        self.chk_largest = QCheckBox("Keep Largest Object Only (Ignore Poop)")
        self.chk_largest.setChecked(True)
        
        self.btn_auto_current = QPushButton("Auto Mask Current (Test)")
        self.btn_auto_current.clicked.connect(self.apply_auto_mask_current)
        self.btn_auto_all = QPushButton("Auto Mask ALL (Parallel)")
        self.btn_auto_all.setStyleSheet("background-color: #6daee2;")
        self.btn_auto_all.clicked.connect(self.apply_auto_mask_all)
        l3.addWidget(QLabel("Thresh:"))
        l3.addWidget(self.spin_thresh)
        l3.addWidget(self.chk_invert)
        l3.addWidget(self.chk_largest)
        l3.addWidget(self.btn_auto_current)
        l3.addWidget(self.btn_auto_all)
        l3.addStretch()
        gb3.setLayout(l3)
        
        # Step 4: Manual Review
        gb4 = QGroupBox("Step 4: Manual Checking & Post-process")
        l4 = QHBoxLayout()
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 100)
        self.spin_brush.setValue(self.brush_size)
        self.spin_brush.valueChanged.connect(self.change_brush)
        self.chk_erase = QCheckBox("Erase Mode (E / W)")
        self.chk_erase.stateChanged.connect(self.toggle_erase)
        
        self.btn_clean_all = QPushButton("Clean Artifacts (Parallel)")
        self.btn_clean_all.setStyleSheet("background-color: #e06666; color: white;")
        self.btn_clean_all.clicked.connect(self.clean_all_artifacts)
        
        l4.addWidget(QLabel("Brush Size:"))
        l4.addWidget(self.spin_brush)
        l4.addWidget(self.chk_erase)
        l4.addWidget(self.btn_clean_all)
        l4.addStretch()
        gb4.setLayout(l4)
        
        # Step 5: Export / Data Extraction
        gb5 = QGroupBox("Step 5: Final Output & Data Extraction")
        l5 = QHBoxLayout()
        self.btn_export_npy = QPushButton("Compile to Array (.npy)")
        self.btn_export_npy.setStyleSheet("background-color: #8e7cc3; color: white; font-weight: bold;")
        self.btn_export_npy.clicked.connect(self.export_numpy)
        
        self.btn_export_avi = QPushButton("Render Labeled Video (.avi)")
        self.btn_export_avi.setStyleSheet("background-color: #93c47d;")
        self.btn_export_avi.clicked.connect(self.export_video)
        
        l5.addWidget(self.btn_export_npy)
        l5.addWidget(self.btn_export_avi)
        l5.addStretch()
        gb5.setLayout(l5)
        
        h_all_steps = QHBoxLayout()
        v_steps_left = QVBoxLayout()
        v_steps_left.addWidget(gb1)
        v_steps_left.addWidget(gb3)
        v_steps_right = QVBoxLayout()
        v_steps_right.addWidget(gb2)
        v_steps_right.addWidget(gb4)
        h_all_steps.addLayout(v_steps_left)
        h_all_steps.addLayout(v_steps_right)
        
        # Slider is below image
        h_slider = QHBoxLayout()
        self.btn_prev = QPushButton("<")
        self.btn_prev.setFixedWidth(30)
        self.btn_prev.clicked.connect(self.prev_frame)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.slider_moved)
        self.btn_next = QPushButton(">")
        self.btn_next.setFixedWidth(30)
        self.btn_next.clicked.connect(self.next_frame)
        h_slider.addWidget(self.btn_prev)
        h_slider.addWidget(self.slider)
        h_slider.addWidget(self.btn_next)
        
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMouseTracking(True)
        self.lbl_image.mousePressEvent = self.mouse_press
        self.lbl_image.mouseMoveEvent = self.mouse_move
        self.lbl_image.mouseReleaseEvent = self.mouse_release
        
        v_layout.addWidget(self.lbl_image, stretch=1)
        v_layout.addLayout(h_slider)
        v_layout.addLayout(h_all_steps)
        v_layout.addWidget(gb5)
        
        # --- Right Panel: Stimulus Marking ---
        v_right = QVBoxLayout()
        v_right.addWidget(QLabel("<b>Stimulus Marking</b>"))
        
        self.lbl_pending_stim = QLabel("Pending:\n[Start: --]\n[End: --]")
        self.btn_stim_start = QPushButton("Mark START Here")
        self.btn_stim_start.setStyleSheet("background-color: #fce5cd;")
        self.btn_stim_start.clicked.connect(self.mark_stim_start)
        self.btn_stim_end = QPushButton("Mark END Here")
        self.btn_stim_end.setStyleSheet("background-color: #d9ead3;")
        self.btn_stim_end.clicked.connect(self.mark_stim_end)
        self.btn_stim_add = QPushButton("+ Add Stimulus to List")
        self.btn_stim_add.setStyleSheet("font-weight: bold;")
        self.btn_stim_add.clicked.connect(self.save_stim_event)
        self.list_stimulus = QListWidget()
        self.btn_stim_del = QPushButton("- Remove Selected")
        self.btn_stim_del.clicked.connect(self.del_stim_event)
        
        v_right.addWidget(self.lbl_pending_stim)
        v_right.addWidget(self.btn_stim_start)
        v_right.addWidget(self.btn_stim_end)
        v_right.addWidget(self.btn_stim_add)
        v_right.addWidget(self.list_stimulus)
        v_right.addWidget(self.btn_stim_del)
        
        main_h_layout.addLayout(v_layout, stretch=4)
        main_h_layout.addLayout(v_right, stretch=1)
        self.setLayout(main_h_layout)
        
        # Disable buttons until video is loaded
        for btn in [self.btn_set_start, self.btn_set_end, self.btn_calc_baseline, 
                    self.btn_define_arena, self.btn_auto_current, self.btn_auto_all,
                    self.btn_clean_all, self.btn_export_avi, self.btn_export_npy,
                    self.btn_stim_start, self.btn_stim_end, self.btn_stim_add, self.btn_stim_del]:
            btn.setEnabled(False)
        
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
        
        self.analysis_start_frame = 0
        self.analysis_end_frame = self.total_frames
        self.lbl_window.setText(f"Window: 0 -> {self.total_frames}")
        
        # Temporary internal mask folder, real data output is .npy at the end
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.mask_folder = os.path.join(os.path.dirname(self.video_path), f"_{video_name}_internal_cache")
        if not os.path.exists(self.mask_folder):
            os.makedirs(self.mask_folder)
            
        self.stats_file = os.path.join(os.path.dirname(self.video_path), f"{video_name}_density_stats.csv")
        self.arena_file = os.path.join(os.path.dirname(self.video_path), f"{video_name}_arena_bounds.json")
        self.stim_file = os.path.join(os.path.dirname(self.video_path), f"{video_name}_stimulus_events.csv")
        
        if os.path.exists(self.arena_file):
            with open(self.arena_file, "r") as f:
                data = json.load(f)
                self.arena_history = {int(k): v for k, v in data.items()}
        else:
            self.arena_history = {}
            
        self.stimulus_events = []
        if os.path.exists(self.stim_file):
            df = pd.read_csv(self.stim_file)
            self.stimulus_events = df.to_dict('records')
        self.refresh_stim_list()
            
        # Re-enable all step buttons
        for btn in [self.btn_set_start, self.btn_set_end, self.btn_calc_baseline, 
                    self.btn_define_arena, self.btn_auto_current, self.btn_auto_all,
                    self.btn_clean_all, self.btn_export_avi, self.btn_export_npy,
                    self.btn_stim_start, self.btn_stim_end, self.btn_stim_add, self.btn_stim_del]:
            btn.setEnabled(True)
            
        self.read_frame()
        self.update_display()
        
    # -------- Stimulus Tracking --------
    def mark_stim_start(self):
        self.pending_stim_start = self.current_frame_idx
        self.update_stim_label()
        
    def mark_stim_end(self):
        self.pending_stim_end = self.current_frame_idx
        self.update_stim_label()
        
    def update_stim_label(self):
        s = self.pending_stim_start if self.pending_stim_start is not None else "--"
        e = self.pending_stim_end if self.pending_stim_end is not None else "--"
        self.lbl_pending_stim.setText(f"Pending:\n[Start: {s}]\n[End: {e}]")
        
    def save_stim_event(self):
        if self.pending_stim_start is None or self.pending_stim_end is None:
            QMessageBox.warning(self, "Incomplete", "Please mark both Start and End frames first!")
            return
        if self.pending_stim_end < self.pending_stim_start:
            QMessageBox.warning(self, "Invalid", "End frame must be after or equal to Start frame.")
            return
            
        dur = self.pending_stim_end - self.pending_stim_start
        self.stimulus_events.append({'Start': self.pending_stim_start, 'End': self.pending_stim_end, 'Duration': dur})
        
        self.pending_stim_start = None
        self.pending_stim_end = None
        self.update_stim_label()
        
        self.refresh_stim_list()
        self.write_stim_csv()
        
    def del_stim_event(self):
        row = self.list_stimulus.currentRow()
        if row < 0 or row >= len(self.stimulus_events): return
        self.stimulus_events.pop(row)
        self.refresh_stim_list()
        self.write_stim_csv()
        
    def refresh_stim_list(self):
        self.list_stimulus.clear()
        for i, ev in enumerate(self.stimulus_events):
            self.list_stimulus.addItem(f"Evt {i+1}: F{ev['Start']} -> F{ev['End']} ({ev['Duration']} frames)")
            
    def write_stim_csv(self):
        if not self.stim_file: return
        pd.DataFrame(self.stimulus_events).to_csv(self.stim_file, index=False)

    # -------- Analysis Window --------
    def set_start_frame(self):
        self.analysis_start_frame = self.current_frame_idx
        self.lbl_window.setText(f"Window: {self.analysis_start_frame} -> {self.analysis_end_frame}")
        QMessageBox.information(self, "Start Time Set", f"Analysis start boundary set to frame {self.analysis_start_frame}.")
        
    def set_end_frame(self):
        self.analysis_end_frame = self.current_frame_idx
        self.lbl_window.setText(f"Window: {self.analysis_start_frame} -> {self.analysis_end_frame}")
        QMessageBox.information(self, "End Time Set", f"Analysis end boundary set to frame {self.analysis_end_frame}.")
        
    # -------- DATA EXPORT MODULES --------
    def export_numpy(self):
        if self.cap is None: return
        export_path = os.path.splitext(self.video_path)[0] + "_binary_masks.npy"
        
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_export_frames = self.analysis_end_frame - self.analysis_start_frame
        
        self.btn_export_npy.setText("...Compiling Array...")
        QApplication.processEvents()
        
        stack = np.zeros((total_export_frames, h, w), dtype=np.uint8)
        
        for idx, i in enumerate(range(self.analysis_start_frame, self.analysis_end_frame)):
            m_path = os.path.join(self.mask_folder, f"mask_{i:04d}.png")
            if os.path.exists(m_path):
                mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    stack[idx] = np.where(mask > 0, 1, 0)
            
            if idx % 100 == 0:
                self.lbl_frame_info.setText(f"Stacking: {idx}/{total_export_frames}")
                QApplication.processEvents()
                
        np.save(export_path, stack)
        
        self.read_frame()
        self.update_display()
        self.btn_export_npy.setText("Compile to Array (.npy)")
        QMessageBox.information(self, "Export Complete", f"Successfully extracted your dataset!\n\nPath: {export_path}\nShape: {stack.shape}\nData Type: uint8 binary array (1/0)\n\nThis removes any reliance on saved PNG images for your downstream mathematical analysis scripts!")
        
    def export_video(self):
        if self.cap is None: return
        export_path = os.path.splitext(self.video_path)[0] + "_labeled.avi"
        
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(export_path, fourcc, fps, (w, h))
        
        self.btn_export_avi.setText("...Exporting...")
        QApplication.processEvents()
        
        for i in range(self.analysis_start_frame, self.analysis_end_frame):
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
                self.lbl_frame_info.setText(f"Exporting: {i-self.analysis_start_frame}/{self.analysis_end_frame-self.analysis_start_frame}")
                QApplication.processEvents()
                
        out.release()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.read_frame()
        self.update_display()
        self.btn_export_avi.setText("Render Labeled Video (.avi)")
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
        
    # -------- Auto Background Subtraction & Parallel Pipelines --------
    def _merge_stats_parallel(self, flat_results):
        df_new = pd.DataFrame(flat_results)
        if df_new.empty: return
        
        if os.path.exists(self.stats_file):
            df_existing = pd.read_csv(self.stats_file)
            if not df_existing.empty:
                df_existing.set_index('Frame', inplace=True)
                df_new.set_index('Frame', inplace=True)
                df_existing.update(df_new)
                df_combined = pd.concat([df_existing[~df_existing.index.isin(df_new.index)], df_new])
                df_combined.reset_index(inplace=True)
                df_combined.sort_values(by='Frame').to_csv(self.stats_file, index=False)
                return
                
        df_new.sort_values(by='Frame').to_csv(self.stats_file, index=False)

    def calc_baseline(self):
        if self.cap is None: return
        self.btn_calc_baseline.setText("...Calculating...")
        QApplication.processEvents()
        
        frames = []
        step = max(1, (self.analysis_end_frame - self.analysis_start_frame) // 30)
        for i in range(self.analysis_start_frame, self.analysis_end_frame, step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, fr = self.cap.read()
            if ret: frames.append(fr)
                
        if len(frames) > 0:
            self.baseline_bgr = np.median(frames, axis=0).astype(np.uint8)
            QMessageBox.information(self, "Success", "Baseline median computed successfully!")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.btn_calc_baseline.setText("Calc Baseline")
        
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
        
        if self.chk_largest.isChecked():
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                closed.fill(0)
                cv2.drawContours(closed, [largest], -1, 255, -1)
                
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
            
        reply = QMessageBox.question(self, "Confirm", f"Process frames {self.analysis_start_frame} to {self.analysis_end_frame} in Parallel? Existing masks in this range will be overwritten.", QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes: return
        
        self.btn_auto_all.setText("...Processing in Parallel...")
        QApplication.processEvents()
        
        n_cpus = multiprocessing.cpu_count()
        total_len = self.analysis_end_frame - self.analysis_start_frame
        chunk_size = max(1, total_len // (n_cpus * 2))
        
        chunks = []
        for i in range(self.analysis_start_frame, self.analysis_end_frame, chunk_size):
            chunks.append((self.video_path, i, min(self.analysis_end_frame, i + chunk_size), 
                           self.baseline_bgr, self.spin_thresh.value(), 
                           self.chk_invert.isChecked(), self.arena_history, self.mask_folder, self.chk_largest.isChecked()))
                           
        with multiprocessing.Pool(n_cpus) as pool:
            results = pool.map(process_video_chunk_auto_mask, chunks)
            
        flat_results = [item for sublist in results for item in sublist]
        self._merge_stats_parallel(flat_results)
        
        self.btn_auto_all.setText("Auto Mask ALL (Parallel)")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.read_frame()
        self.update_display()
        QMessageBox.information(self, "Done", f"Time Window Processed instantly across {n_cpus} CPUs!")
        
    def clean_all_artifacts(self):
        reply = QMessageBox.question(self, "Clean Global Artifacts", "This distributes processing across ALL CPUs to rigidly DELETE everything except the absolute single longest solid shape per frame.\n\nAre you sure you want to proceed?", QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes: return
        
        self.btn_clean_all.setText("...Cleaning in Parallel...")
        QApplication.processEvents()
        
        n_cpus = multiprocessing.cpu_count()
        total_len = self.analysis_end_frame - self.analysis_start_frame
        chunk_size = max(1, total_len // (n_cpus * 2))
        
        chunks = []
        for i in range(self.analysis_start_frame, self.analysis_end_frame, chunk_size):
            chunks.append((i, min(self.analysis_end_frame, i + chunk_size), self.mask_folder))
            
        with multiprocessing.Pool(n_cpus) as pool:
            results = pool.map(process_video_chunk_clean_artifacts, chunks)
            
        flat_results = [item for sublist in results for item in sublist]
        self._merge_stats_parallel(flat_results)
                    
        self.btn_clean_all.setText("Clean Artifacts (Parallel)")
        self.read_frame()
        self.update_display()
        QMessageBox.information(self, "Completed", f"Successfully purged all disjointed artifacts/poop in Parallel across {n_cpus} CPUs! Only the largest primary target remains.")
        
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
