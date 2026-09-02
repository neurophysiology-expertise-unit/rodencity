import cv2
import os
import tempfile
import pandas as pd
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QSlider, QListWidget, QGroupBox, QComboBox,
    QMessageBox, QProgressDialog, QSizePolicy, QDoubleSpinBox,
    QApplication,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


BEHAVIOR_PRESETS = ["seizure", "grooming", "rearing", "freezing", "exploration", "other"]

LOCAL_CACHE_DIR = Path(tempfile.gettempdir()) / "rodencity_cache"


class _CopyWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, src: str, dst: str):
        super().__init__()
        self.src = src
        self.dst = dst
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            src_size = os.path.getsize(self.src)
            chunk = 1024 * 1024  # 1 MB per chunk
            copied = 0
            with open(self.src, "rb") as fin, open(self.dst, "wb") as fout:
                while True:
                    if self._cancelled:
                        return
                    data = fin.read(chunk)
                    if not data:
                        break
                    fout.write(data)
                    copied += len(data)
                    self.progress.emit(int(copied * 100 / src_size) if src_size else 100)
            self.finished.emit(self.dst)
        except Exception as exc:
            self.error.emit(str(exc))


class BehaviorAnnotator(QWidget):
    """Lightweight behavioral event annotator with local video caching."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rodencity — Behavior Annotator")
        self.resize(1200, 720)

        self.source_path: str | None = None
        self.local_path: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0
        self.is_playing = False

        self.pending_start: int | None = None
        self.pending_end: int | None = None
        self.annotations: list[dict] = []
        self.output_csv: str | None = None

        self._play_timer = QTimer()
        self._play_timer.timeout.connect(self._advance_frame)

        self.setFocusPolicy(Qt.StrongFocus)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout()

        # ---- Left column: video display + transport controls ----
        v_left = QVBoxLayout()

        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background: #111;")
        self.lbl_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v_left.addWidget(self.lbl_image, stretch=1)

        h_slider = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.lbl_time = QLabel("0:00.00 / 0:00.00")
        h_slider.addWidget(self.slider)
        h_slider.addWidget(self.lbl_time)
        v_left.addLayout(h_slider)

        h_transport = QHBoxLayout()
        self.btn_load = QPushButton("Load Video")
        self.btn_load.clicked.connect(self._load_video)

        self.btn_prev = QPushButton("< Frame")
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._prev_frame)

        self.btn_play = QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.btn_play.setFixedWidth(70)
        self.btn_play.clicked.connect(self._toggle_play)

        self.btn_next = QPushButton("Frame >")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._next_frame)

        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 8.0)
        self.spin_speed.setSingleStep(0.25)
        self.spin_speed.setValue(1.0)
        self.spin_speed.setDecimals(2)
        self.spin_speed.setPrefix("×")
        self.spin_speed.setEnabled(False)
        self.spin_speed.valueChanged.connect(self._update_timer_interval)

        h_transport.addWidget(self.btn_load)
        h_transport.addStretch()
        h_transport.addWidget(self.btn_prev)
        h_transport.addWidget(self.btn_play)
        h_transport.addWidget(self.btn_next)
        h_transport.addStretch()
        h_transport.addWidget(QLabel("Speed:"))
        h_transport.addWidget(self.spin_speed)
        v_left.addLayout(h_transport)

        self.lbl_status = QLabel("Select a video to begin. Remote/Samba videos are copied locally first.")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: gray; font-size: 10px;")
        v_left.addWidget(self.lbl_status)

        root.addLayout(v_left, stretch=4)

        # ---- Right column: annotation panel ----
        v_right = QVBoxLayout()
        v_right.setSpacing(6)

        gb_label = QGroupBox("Behavior Label")
        l_label = QVBoxLayout()
        l_label.addWidget(QLabel("Label (choose preset or type custom):"))
        self.combo_label = QComboBox()
        self.combo_label.addItems(BEHAVIOR_PRESETS)
        self.combo_label.setEditable(True)
        self.combo_label.setInsertPolicy(QComboBox.NoInsert)
        l_label.addWidget(self.combo_label)
        gb_label.setLayout(l_label)
        v_right.addWidget(gb_label)

        gb_mark = QGroupBox("Mark Interval")
        l_mark = QVBoxLayout()

        self.lbl_pending = QLabel("Start: --\nEnd:   --")
        self.lbl_pending.setStyleSheet("font-family: monospace; padding: 4px;")

        self.btn_mark_start = QPushButton("Mark Start  [S]")
        self.btn_mark_start.setEnabled(False)
        self.btn_mark_start.clicked.connect(self._mark_start)

        self.btn_mark_end = QPushButton("Mark End  [E]")
        self.btn_mark_end.setEnabled(False)
        self.btn_mark_end.clicked.connect(self._mark_end)

        self.btn_add = QPushButton("+ Add Annotation  [Enter]")
        self.btn_add.setEnabled(False)
        self.btn_add.clicked.connect(self._add_annotation)

        l_mark.addWidget(self.lbl_pending)
        l_mark.addWidget(self.btn_mark_start)
        l_mark.addWidget(self.btn_mark_end)
        l_mark.addWidget(self.btn_add)
        gb_mark.setLayout(l_mark)
        v_right.addWidget(gb_mark)

        l_hint = QLabel(
            "Shortcuts:\n"
            "  Space — play / pause\n"
            "  A / ← — previous frame\n"
            "  D / → — next frame\n"
            "  S — mark start\n"
            "  E — mark end\n"
            "  Enter — add annotation"
        )
        l_hint.setStyleSheet("font-size: 10px; color: gray; padding: 4px;")
        v_right.addWidget(l_hint)

        gb_list = QGroupBox("Saved Annotations")
        l_list = QVBoxLayout()
        self.list_annotations = QListWidget()
        self.btn_delete = QPushButton("- Remove Selected")
        self.btn_delete.setEnabled(False)
        self.btn_delete.clicked.connect(self._delete_annotation)
        self.lbl_csv = QLabel("CSV: —")
        self.lbl_csv.setWordWrap(True)
        self.lbl_csv.setStyleSheet("font-size: 9px; color: gray;")
        l_list.addWidget(self.list_annotations)
        l_list.addWidget(self.btn_delete)
        l_list.addWidget(self.lbl_csv)
        gb_list.setLayout(l_list)
        v_right.addWidget(gb_list, stretch=1)

        root.addLayout(v_right, stretch=1)
        self.setLayout(root)

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.cap:
                self._toggle_play()
        elif event.key() in (Qt.Key_A, Qt.Key_Left):
            self._prev_frame()
        elif event.key() in (Qt.Key_D, Qt.Key_Right):
            self._next_frame()
        elif event.key() == Qt.Key_S:
            if self.cap:
                self._mark_start()
        elif event.key() == Qt.Key_E:
            if self.cap:
                self._mark_end()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.cap:
                self._add_annotation()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Video loading & caching
    # ------------------------------------------------------------------
    def _load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.avi *.mp4 *.mkv *.mov)"
        )
        if not path:
            return

        self.source_path = path
        LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        local_dst = str(LOCAL_CACHE_DIR / Path(path).name)

        # If already cached with matching size, skip copy
        try:
            if os.path.exists(local_dst) and os.path.getsize(local_dst) == os.path.getsize(path):
                self._open_local(local_dst)
                return
        except OSError:
            pass

        self._copy_worker = _CopyWorker(path, local_dst)
        self._progress_dlg = QProgressDialog(
            f"Copying to local cache…\n{Path(path).name}", "Cancel", 0, 100, self
        )
        self._progress_dlg.setWindowTitle("Loading")
        self._progress_dlg.setWindowModality(Qt.WindowModal)
        self._progress_dlg.canceled.connect(self._copy_worker.cancel)
        self._progress_dlg.setValue(0)

        self._copy_worker.progress.connect(self._progress_dlg.setValue)
        self._copy_worker.finished.connect(self._on_copy_done)
        self._copy_worker.error.connect(self._on_copy_error)
        self._copy_worker.start()
        self._progress_dlg.exec_()

    def _on_copy_done(self, local_path: str):
        self._progress_dlg.close()
        self._open_local(local_path)

    def _on_copy_error(self, msg: str):
        self._progress_dlg.close()
        QMessageBox.critical(self, "Copy Failed", f"Could not copy video to local cache:\n{msg}")

    def _open_local(self, local_path: str):
        if self.cap:
            self.cap.release()

        self.local_path = local_path
        self.cap = cv2.VideoCapture(local_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        if self.total_frames == 0:
            QMessageBox.critical(self, "Error", "Could not read any frames from this video.")
            self.cap.release()
            self.cap = None
            return

        self.current_frame_idx = 0
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)

        for w in [self.btn_prev, self.btn_play, self.btn_next, self.spin_speed,
                  self.btn_mark_start, self.btn_mark_end, self.btn_add, self.btn_delete]:
            w.setEnabled(True)

        # Output CSV lives next to the source video (on the network share)
        src_dir = Path(self.source_path).parent
        stem = Path(self.source_path).stem
        self.output_csv = str(src_dir / f"{stem}_behavior_annotations.csv")
        self.lbl_csv.setText(f"CSV: {self.output_csv}")

        # Load existing annotations for this video if any
        self.annotations = []
        if os.path.exists(self.output_csv):
            try:
                df = pd.read_csv(self.output_csv)
                self.annotations = df.to_dict("records")
            except Exception:
                pass
        self._refresh_list()

        self.pending_start = None
        self.pending_end = None
        self._update_pending_label()
        self._render_frame()

        dur = self.total_frames / self.fps
        self.lbl_status.setText(
            f"{Path(self.source_path).name}  |  "
            f"{self.total_frames} frames  |  {self.fps:.2f} fps  |  "
            f"Duration: {self._fmt_time(dur)}  |  "
            f"Cache: {Path(local_path).name}"
        )

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    def _toggle_play(self):
        if not self.cap:
            return
        if self.is_playing:
            self._play_timer.stop()
            self.is_playing = False
            self.btn_play.setText("Play")
        else:
            if self.current_frame_idx >= self.total_frames - 1:
                self.current_frame_idx = 0
            self._update_timer_interval()
            self._play_timer.start()
            self.is_playing = True
            self.btn_play.setText("Pause")

    def _update_timer_interval(self):
        if self.fps > 0:
            ms = max(1, int(1000 / (self.fps * self.spin_speed.value())))
            self._play_timer.setInterval(ms)

    def _advance_frame(self):
        if self.current_frame_idx >= self.total_frames - 1:
            self._play_timer.stop()
            self.is_playing = False
            self.btn_play.setText("Play")
            return
        self.current_frame_idx += 1
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_idx)
        self.slider.blockSignals(False)
        self._render_frame()

    def _on_slider(self, val: int):
        if not self.cap:
            return
        self.current_frame_idx = val
        self._render_frame()

    def _prev_frame(self):
        if not self.cap or self.current_frame_idx <= 0:
            return
        self.current_frame_idx -= 1
        self.slider.setValue(self.current_frame_idx)

    def _next_frame(self):
        if not self.cap or self.current_frame_idx >= self.total_frames - 1:
            return
        self.current_frame_idx += 1
        self.slider.setValue(self.current_frame_idx)

    def _render_frame(self):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = self._draw_overlays(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qi)

        lw, lh = self.lbl_image.width(), self.lbl_image.height()
        if lw > 0 and lh > 0:
            pm = pm.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.lbl_image.setPixmap(pm)

        cur_sec = self.current_frame_idx / self.fps
        tot_sec = self.total_frames / self.fps
        self.lbl_time.setText(
            f"{self._fmt_time(cur_sec)} / {self._fmt_time(tot_sec)}  [F{self.current_frame_idx}]"
        )

    def _draw_overlays(self, frame):
        """Highlight active annotation intervals and pending markers."""
        bar_h = 8
        for ann in self.annotations:
            if ann["Start_Frame"] <= self.current_frame_idx <= ann["End_Frame"]:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_h), (0, 200, 80), -1)
                cv2.putText(
                    frame, ann["Label"], (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA,
                )
                break

        if self.pending_start is not None and self.current_frame_idx == self.pending_start:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_h), (0, 180, 255), -1)
        if self.pending_end is not None and self.current_frame_idx == self.pending_end:
            fh = frame.shape[0]
            cv2.rectangle(frame, (0, fh - bar_h), (frame.shape[1], fh), (255, 100, 0), -1)

        return frame

    @staticmethod
    def _fmt_time(sec: float) -> str:
        m = int(sec // 60)
        s = sec % 60
        return f"{m}:{s:05.2f}"

    # ------------------------------------------------------------------
    # Annotation logic
    # ------------------------------------------------------------------
    def _mark_start(self):
        self.pending_start = self.current_frame_idx
        self._update_pending_label()

    def _mark_end(self):
        self.pending_end = self.current_frame_idx
        self._update_pending_label()

    def _update_pending_label(self):
        def fmt(f):
            if f is None:
                return "--"
            return f"F{f}  ({self._fmt_time(f / self.fps)})"

        self.lbl_pending.setText(f"Start: {fmt(self.pending_start)}\nEnd:   {fmt(self.pending_end)}")

    def _add_annotation(self):
        if self.pending_start is None or self.pending_end is None:
            QMessageBox.warning(self, "Incomplete", "Mark both a Start and End frame first.")
            return
        if self.pending_end < self.pending_start:
            QMessageBox.warning(self, "Invalid", "End frame must be at or after Start frame.")
            return

        label = self.combo_label.currentText().strip()
        if not label:
            QMessageBox.warning(self, "No Label", "Enter a behavior label before adding.")
            return

        start_sec = round(self.pending_start / self.fps, 3)
        end_sec = round(self.pending_end / self.fps, 3)

        self.annotations.append({
            "Label": label,
            "Start_Frame": self.pending_start,
            "End_Frame": self.pending_end,
            "Start_Time_Sec": start_sec,
            "End_Time_Sec": end_sec,
            "Duration_Sec": round(end_sec - start_sec, 3),
        })

        self.pending_start = None
        self.pending_end = None
        self._update_pending_label()
        self._refresh_list()
        self._save_csv()

    def _delete_annotation(self):
        row = self.list_annotations.currentRow()
        if 0 <= row < len(self.annotations):
            self.annotations.pop(row)
            self._refresh_list()
            self._save_csv()

    def _refresh_list(self):
        self.list_annotations.clear()
        for i, a in enumerate(self.annotations):
            self.list_annotations.addItem(
                f"{i+1}. [{a['Label']}]  "
                f"F{a['Start_Frame']} → F{a['End_Frame']}  "
                f"({a['Start_Time_Sec']}s – {a['End_Time_Sec']}s, "
                f"{a['Duration_Sec']}s)"
            )

    def _save_csv(self):
        if not self.output_csv:
            return
        try:
            pd.DataFrame(self.annotations).to_csv(self.output_csv, index=False)
        except Exception as exc:
            QMessageBox.warning(self, "Save Failed", f"Could not write annotation CSV:\n{exc}")

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._play_timer.stop()
        if self.cap:
            self.cap.release()
        super().closeEvent(event)
