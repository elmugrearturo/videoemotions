from videowindow import VentanaDeVideo
from PyQt5.QtWidgets import QApplication

import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = VentanaDeVideo()
    ventana.show()
    ventana.setWindowTitle("Demo Tesis")
    ventana.start()
    sys.exit(app.exec_())

