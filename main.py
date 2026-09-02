import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt
from gui.annotator import VideoAnnotator
from gui.behavior_annotator import BehaviorAnnotator


def _pick_mode() -> int:
    """Return 1 for Analysis Pipeline, 2 for Behavior Annotator, 0 to quit."""
    dlg = QDialog()
    dlg.setWindowTitle("Rodencity")
    dlg.setFixedWidth(400)

    layout = QVBoxLayout()
    layout.setSpacing(12)

    title = QLabel("<b>Rodencity</b>")
    title.setAlignment(Qt.AlignCenter)
    title.setStyleSheet("font-size: 18px; padding: 8px;")
    layout.addWidget(title)

    layout.addWidget(QLabel("Select a tool to open:"))

    btn_analysis = QPushButton(
        "Analysis Pipeline\n"
        "Background subtraction, arena masking, density tracking"
    )
    btn_analysis.setFixedHeight(60)
    btn_analysis.clicked.connect(lambda: dlg.done(1))

    btn_behavior = QPushButton(
        "Behavior Annotator\n"
        "Label behavioral events (seizure, grooming, …) with time intervals"
    )
    btn_behavior.setFixedHeight(60)
    btn_behavior.clicked.connect(lambda: dlg.done(2))

    layout.addWidget(btn_analysis)
    layout.addWidget(btn_behavior)
    dlg.setLayout(layout)

    return dlg.exec_()


def main():
    app = QApplication(sys.argv)

    mode = _pick_mode()
    if mode == 1:
        window = VideoAnnotator()
    elif mode == 2:
        window = BehaviorAnnotator()
    else:
        sys.exit(0)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
