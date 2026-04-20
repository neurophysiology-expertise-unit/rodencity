import cv2
import numpy as np
import pandas as pd
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QSlider, QSpinBox, QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

class VideoAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mouse Video Annotator (with Auto-Detect)")
        
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
        
        self.mask_folder = None
        self.stats_file = None
        
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
        self.btn_calc_baseline = QPushButton("2. Calc Baseline (Median)")
        self.btn_calc_baseline.clicked.connect(self.calc_baseline)
        
        self.spin_thresh = QSpinBox()
        self.spin_thresh.setRange(1, 255)
        self.spin_thresh.setValue(30)
        self.spin_thresh.setToolTip("Lower = picks up faint things. Higher = blocks shadow.")
        
        self.btn_auto_current = QPushButton("Auto Mask Current")
        self.btn_auto_current.clicked.connect(self.apply_auto_mask_current)
        
        self.btn_auto_all = QPushButton("Auto Mask ALL")
        self.btn_auto_all.setStyleSheet("background-color: #6daee2;")
        self.btn_auto_all.clicked.connect(self.apply_auto_mask_all)
        
        h_auto.addWidget(self.btn_calc_baseline)
        h_auto.addWidget(QLabel("Detection Threshold:"))
        h_auto.addWidget(self.spin_thresh)
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
        
        self.btn_save = QPushButton("Save Mask")
        self.btn_save.clicked.connect(self.save_current_mask)
        self.btn_clear = QPushButton("Clear Mask")
        self.btn_clear.clicked.connect(self.clear_mask)
        self.btn_interpolate = QPushButton("Interpolate Gap")
        self.btn_interpolate.clicked.connect(self.interpolate_gap)
        
        h_ctrl.addWidget(QLabel("Brush Size:"))
        h_ctrl.addWidget(self.spin_brush)
        h_ctrl.addWidget(self.chk_erase)
        h_ctrl.addWidget(self.btn_clear)
        h_ctrl.addWidget(self.btn_interpolate)
        h_ctrl.addWidget(self.btn_save)
        
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
        self.read_frame()
        self.update_display()
        
    # -------- Auto Background Subtraction --------
    def calc_baseline(self):
        if self.cap is None: return
        self.btn_calc_baseline.setText("...Calculating...")
        QApplication.processEvents()
        
        frames = []
        step = max(1, self.total_frames // 30) # sample 30 frames
        for i in range(0, self.total_frames, step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, fr = self.cap.read()
            if ret: frames.append(fr)
                
        if len(frames) > 0:
            self.baseline_bgr = np.median(frames, axis=0).astype(np.uint8)
            QMessageBox.information(self, "Success", "Baseline median computed successfully!")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.btn_calc_baseline.setText("Baseline Ready")
        
    def auto_mask_frame(self, frame_img):
        if self.baseline_bgr is None: return None
        
        diff = cv2.absdiff(frame_img, self.baseline_bgr)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, self.spin_thresh.value(), 255, cv2.THRESH_BINARY)
        
        # morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return closed
        
    def apply_auto_mask_current(self):
        if self.baseline_bgr is None:
            QMessageBox.warning(self, "Warning", "Please calculate baseline first!")
            return
        m = self.auto_mask_frame(self.frame_bgr)
        if m is not None:
            self.mask = m
            self.save_current_mask()
            self.update_display()
            
    def apply_auto_mask_all(self):
        if self.baseline_bgr is None:
            QMessageBox.warning(self, "Warning", "Please calculate baseline first!")
            return
            
        reply = QMessageBox.question(self, "Confirm", "Process all frames? Existing masks will be overwritten.", QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes: return
        
        for i in range(self.total_frames):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, fr = self.cap.read()
            if ret:
                m = self.auto_mask_frame(fr)
                cv2.imwrite(os.path.join(self.mask_folder, f"mask_{i:04d}.png"), m)
                self.update_stats(i, m)
                
                if i % 10 == 0:
                    self.lbl_frame_info.setText(f"Processing: {i}/{self.total_frames}")
                    QApplication.processEvents()
                    
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.read_frame()
        self.update_display()
        QMessageBox.information(self, "Done", "All frames processed and saved!")
        
    # -------- Navigation --------
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
        
    # -------- Utilities --------
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
        overlay[self.mask > 0] = [0, 0, 255]
        blended = cv2.addWeighted(overlay, 0.4, self.frame_bgr, 0.6, 0)
        
        rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.lbl_image.setPixmap(QPixmap.fromImage(qimg))
        
    def draw_on_mask(self, pos):
        if self.mask is None or self.frame_bgr is None: return
        cv2.circle(self.mask, (pos.x(), pos.y()), self.brush_size, 0 if self.is_erasing else 255, -1)
        self.update_display()
        
    def mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.draw_on_mask(event.pos())
    def mouse_move(self, event):
        if self.is_drawing: self.draw_on_mask(event.pos())
    def mouse_release(self, event):
        if event.button() == Qt.LeftButton:
            self.is_drawing = False
            self.save_current_mask()
