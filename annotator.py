import cv2
import numpy as np
import pandas as pd
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QSlider, QSpinBox, QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

class VideoAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mouse Video Annotator")
        
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        
        self.frame_bgr = None
        self.mask = None
        
        self.brush_size = 10
        self.is_erasing = False
        self.is_drawing = False
        
        self.mask_folder = None
        self.stats_file = None
        
        self.setFocusPolicy(Qt.StrongFocus) # To capture keyboard events
        self.init_ui()
        
    def init_ui(self):
        v_layout = QVBoxLayout()
        h_ctrl = QHBoxLayout()
        
        self.btn_load = QPushButton("1. Load Video")
        self.btn_load.clicked.connect(self.load_video)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.slider_moved)
        
        self.lbl_frame_info = QLabel("Frame: 0/0")
        
        # Navigation Buttons
        self.btn_prev = QPushButton("< Prev (A)")
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next = QPushButton("Next (D) >")
        self.btn_next.clicked.connect(self.next_frame)
        
        h_nav = QHBoxLayout()
        h_nav.addWidget(self.btn_prev)
        h_nav.addWidget(self.slider)
        h_nav.addWidget(self.btn_next)
        h_nav.addWidget(self.lbl_frame_info)
        
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 100)
        self.spin_brush.setValue(self.brush_size)
        self.spin_brush.valueChanged.connect(self.change_brush)
        
        self.chk_erase = QCheckBox("Erase Mode")
        self.chk_erase.stateChanged.connect(self.toggle_erase)
        
        self.btn_save = QPushButton("Save Current Mask")
        self.btn_save.clicked.connect(self.save_current_mask)
        
        self.btn_clear = QPushButton("Clear Mask")
        self.btn_clear.clicked.connect(self.clear_mask)
        
        self.btn_interpolate = QPushButton("Interpolate Gap")
        self.btn_interpolate.clicked.connect(self.interpolate_gap)
        self.btn_interpolate.setToolTip("Interpolate masks between the last saved frame and the current frame.")
        
        h_ctrl.addWidget(self.btn_load)
        h_ctrl.addWidget(QLabel("Brush Size:"))
        h_ctrl.addWidget(self.spin_brush)
        h_ctrl.addWidget(self.chk_erase)
        h_ctrl.addWidget(self.btn_clear)
        h_ctrl.addWidget(self.btn_save)
        h_ctrl.addWidget(self.btn_interpolate)
        
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMouseTracking(True)
        
        self.lbl_image.mousePressEvent = self.mouse_press
        self.lbl_image.mouseMoveEvent = self.mouse_move
        self.lbl_image.mouseReleaseEvent = self.mouse_release
        
        v_layout.addWidget(self.lbl_image)
        v_layout.addLayout(h_nav)
        v_layout.addLayout(h_ctrl)
        
        self.setLayout(v_layout)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.prev_frame()
        elif event.key() == Qt.Key_D:
            self.next_frame()
        else:
            super().keyPressEvent(event)
            
    def change_brush(self, val):
        self.brush_size = val
        
    def toggle_erase(self, state):
        self.is_erasing = state == Qt.Checked
        
    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.avi *.mp4)")
        if not path: return
            
        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.total_frames == 0:
            QMessageBox.warning(self, "Error", "Could not load video frames.")
            return
            
        self.current_frame_idx = 0
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
        
    def prev_frame(self):
        val = max(0, self.current_frame_idx - 1)
        self.slider.setValue(val)
        
    def next_frame(self):
        val = min(self.total_frames - 1, self.current_frame_idx + 1)
        self.slider.setValue(val)
        
    def slider_moved(self, val):
        self.save_current_mask()
        self.current_frame_idx = val
        self.read_frame()
        self.update_display()
        
    def read_frame(self):
        if self.cap is None: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.frame_bgr = frame
            mask_path = os.path.join(self.mask_folder, f"mask_{self.current_frame_idx:04d}.png")
            if os.path.exists(mask_path):
                self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                # If no mask exists, keep the previous mask shape if it exists (superimpose)
                if self.mask is None or self.mask.shape[:2] != self.frame_bgr.shape[:2]:
                    self.mask = np.zeros(self.frame_bgr.shape[:2], dtype=np.uint8)
                else:
                    self.mask = self.mask.copy()
        else:
            self.frame_bgr = None
            
        self.lbl_frame_info.setText(f"Frame: {self.current_frame_idx}/{self.total_frames - 1}")
        
    def clear_mask(self):
        if self.mask is not None:
            self.mask.fill(0)
            self.update_display()
            
    def save_current_mask(self):
        if self.mask is None: return
        mask_path = os.path.join(self.mask_folder, f"mask_{self.current_frame_idx:04d}.png")
        cv2.imwrite(mask_path, self.mask)
        self.update_stats(self.current_frame_idx, self.mask)
        
    def interpolate_gap(self):
        if self.current_frame_idx == 0 or self.mask is None: return
        
        # Search backwards for the last saved mask on disk
        prev_idx = -1
        for i in range(self.current_frame_idx - 1, -1, -1):
            if os.path.exists(os.path.join(self.mask_folder, f"mask_{i:04d}.png")):
                prev_idx = i
                break
                
        if prev_idx == -1:
            QMessageBox.information(self, "Info", "No previous mask found on disk to interpolate from.")
            return
            
        mask_prev = cv2.imread(os.path.join(self.mask_folder, f"mask_{prev_idx:04d}.png"), cv2.IMREAD_GRAYSCALE)
        mask_curr = self.mask
        
        M_prev = cv2.moments(mask_prev)
        M_curr = cv2.moments(mask_curr)
        
        cx_p = M_prev['m10'] / M_prev['m00'] if M_prev['m00'] > 0 else 0
        cy_p = M_prev['m01'] / M_prev['m00'] if M_prev['m00'] > 0 else 0
        
        cx_c = M_curr['m10'] / M_curr['m00'] if M_curr['m00'] > 0 else 0
        cy_c = M_curr['m01'] / M_curr['m00'] if M_curr['m00'] > 0 else 0
        
        total_steps = self.current_frame_idx - prev_idx
        
        for step in range(1, total_steps):
            target_idx = prev_idx + step
            w = step / total_steps
            
            dx = int((cx_c - cx_p) * w)
            dy = int((cy_c - cy_p) * w)
            
            # Translate the mask shape from prev towards current by dx, dy
            # We warp the old mask 
            M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(mask_prev, M_trans, (mask_prev.shape[1], mask_prev.shape[0]))
            
            mask_path = os.path.join(self.mask_folder, f"mask_{target_idx:04d}.png")
            cv2.imwrite(mask_path, shifted)
            self.update_stats(target_idx, shifted)
            
        QMessageBox.information(self, "Success", f"Interpolated shape over {total_steps - 1} frames.")
        
    def update_stats(self, frame_idx, mask_array):
        is_painted = (mask_array > 0)
        mean_val = np.mean(is_painted)
        std_val = np.std(is_painted)
        
        data = []
        if os.path.exists(self.stats_file):
            df = pd.read_csv(self.stats_file)
            data = df.to_dict('records')
            
        found = False
        for row in data:
            if row['Frame'] == frame_idx:
                row['Mean_Density'] = mean_val
                row['Std_Density'] = std_val
                found = True
                break
        if not found:
            data.append({'Frame': frame_idx, 'Mean_Density': mean_val, 'Std_Density': std_val})
            
        df_new = pd.DataFrame(data).sort_values(by='Frame')
        df_new.to_csv(self.stats_file, index=False)
        
    def update_display(self):
        if self.frame_bgr is None: return
            
        overlay = self.frame_bgr.copy()
        overlay[self.mask > 0] = [0, 0, 255]
        
        blended = cv2.addWeighted(overlay, 0.4, self.frame_bgr, 0.6, 0)
        
        rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.lbl_image.setPixmap(QPixmap.fromImage(qimg))
        
    def draw_on_mask(self, pos):
        if self.mask is None or self.frame_bgr is None: return
            
        x, y = pos.x(), pos.y()
        color = 0 if self.is_erasing else 255
        cv2.circle(self.mask, (x, y), self.brush_size, color, -1)
        self.update_display()

    def mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.draw_on_mask(event.pos())
            
    def mouse_move(self, event):
        if self.is_drawing:
            self.draw_on_mask(event.pos())
            
    def mouse_release(self, event):
        if event.button() == Qt.LeftButton:
            self.is_drawing = False
            self.save_current_mask()
