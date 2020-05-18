from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter

class ContenedorDeImagen(QWidget):
    def __init__(self, parent=None):
        super(ContenedorDeImagen, self).__init__(parent)
        self.imagen = None

    def dibujar(self, imagen):
        self.imagen = imagen
        self.setMinimumSize(imagen.size())
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.imagen:
            qp.drawImage(QPoint(0, 0), self.imagen)
        qp.end()

