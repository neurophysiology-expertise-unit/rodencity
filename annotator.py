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
        
        self.init_ui()
        
    def init_ui(self):
        v_layout = QVBoxLayout()
        h_ctrl = QHBoxLayout()
        
        self.btn_load = QPushButton("Load Video")
        self.btn_load.clicked.connect(self.load_video)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.slider_moved)
        
        self.lbl_frame_info = QLabel("Frame: 0/0")
        
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 100)
        self.spin_brush.setValue(self.brush_size)
        self.spin_brush.valueChanged.connect(self.change_brush)
        
        self.chk_erase = QCheckBox("Erase Mode")
        self.chk_erase.stateChanged.connect(self.toggle_erase)
        
        h_ctrl.addWidget(self.btn_load)
        h_ctrl.addWidget(QLabel("Brush Size:"))
        h_ctrl.addWidget(self.spin_brush)
        h_ctrl.addWidget(self.chk_erase)
        
        h_slider = QHBoxLayout()
        h_slider.addWidget(self.slider)
        h_slider.addWidget(self.lbl_frame_info)
        
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMouseTracking(True)
        
        self.lbl_image.mousePressEvent = self.mouse_press
        self.lbl_image.mouseMoveEvent = self.mouse_move
        self.lbl_image.mouseReleaseEvent = self.mouse_release
        
        v_layout.addWidget(self.lbl_image)
        v_layout.addLayout(h_ctrl)
        v_layout.addLayout(h_slider)
        
        self.setLayout(v_layout)
        
    def change_brush(self, val):
        self.brush_size = val
        
    def toggle_erase(self, state):
        self.is_erasing = state == Qt.Checked
        
    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.avi *.mp4)")
        if not path:
            return
            
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
        
    def slider_moved(self, val):
        if self.mask is not None:
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
                self.mask = np.zeros(self.frame_bgr.shape[:2], dtype=np.uint8)
        else:
            self.frame_bgr = None
            self.mask = None
            
        self.lbl_frame_info.setText(f"Frame: {self.current_frame_idx}/{self.total_frames - 1}")
        
    def save_current_mask(self):
        if self.mask is None: return
        # Prevent saving entirely blank masks unless there's a reason, but let's always save
        mask_path = os.path.join(self.mask_folder, f"mask_{self.current_frame_idx:04d}.png")
        cv2.imwrite(mask_path, self.mask)
        self.update_stats()
        
    def update_stats(self):
        # We consider a mask > 0 as painted
        is_painted = (self.mask > 0)
        mean_val = np.mean(is_painted)
        std_val = np.std(is_painted)
        
        data = []
        if os.path.exists(self.stats_file):
            df = pd.read_csv(self.stats_file)
            data = df.to_dict('records')
            
        found = False
        for row in data:
            if row['Frame'] == self.current_frame_idx:
                row['Mean_Density'] = mean_val
                row['Std_Density'] = std_val
                found = True
                break
        if not found:
            data.append({'Frame': self.current_frame_idx, 'Mean_Density': mean_val, 'Std_Density': std_val})
            
        df_new = pd.DataFrame(data).sort_values(by='Frame')
        df_new.to_csv(self.stats_file, index=False)
        
    def update_display(self):
        if self.frame_bgr is None:
            return
            
        overlay = self.frame_bgr.copy()
        # Draw red transparent layer onto the painted pixels
        overlay[self.mask > 0] = [0, 0, 255]
        
        blended = cv2.addWeighted(overlay, 0.4, self.frame_bgr, 0.6, 0)
        
        rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.lbl_image.setPixmap(QPixmap.fromImage(qimg))
        
    def draw_on_mask(self, pos):
        if self.mask is None or self.frame_bgr is None:
            return
            
        x, y = pos.x(), pos.y()
        
        # Prevent drawing outside bounds if possible, opencv circle handles bounds gracefully
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
