import sys
from PyQt5.QtWidgets import QApplication
from annotator import VideoAnnotator

def main():
    app = QApplication(sys.argv)
    window = VideoAnnotator()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
