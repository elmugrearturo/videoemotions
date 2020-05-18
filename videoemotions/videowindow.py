# -*- coding: utf-8 -*-
import sys
import cv2
import threading

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QWidget, QAction, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QFont, QImage

from widgets import ContenedorDeImagen
import videocapture as vc

import queue

buffer_de_imgs = queue.Queue()     

class VentanaDeVideo(QMainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.central = QWidget(self)

        # ORDEN DE LO WIDGETS
        self.vlayout = QVBoxLayout()
        self.displays = QHBoxLayout()
        self.canvas = ContenedorDeImagen(self)    
        self.displays.addWidget(self.canvas)
        self.vlayout.addLayout(self.displays)
        self.central.setLayout(self.vlayout)
        self.setCentralWidget(self.central)

        # MENUS
        self.barra_menus = self.menuBar()
        
        accion_salir = QAction('&Salir', self)
        accion_salir.setShortcut('Ctrl+Q')
        accion_salir.triggered.connect(self.close)
        self.menu_archivo = self.barra_menus.addMenu('&Archivo')
        self.menu_archivo.addAction(accion_salir)

        self.menu_profesor = self.barra_menus.addMenu('&Profesor')
        accion_curso = QAction('&Curso', self)
        accion_curso.triggered.connect(self.curso)
        accion_tema = QAction('&Tema', self)
        accion_tema.triggered.connect(self.tema)
        accion_video = QAction('&Video', self)
        accion_video.triggered.connect(self.video)
        self.menu_profesor.addAction(accion_curso)
        self.menu_profesor.addAction(accion_tema)
        self.menu_profesor.addAction(accion_video)

    # SE EJECUTA AL CREAR LA VENTANA
    def start(self):
        '''Crea un hilo para recuperar la imagen y pone una tasa de 
        actualización para sacar imagenes del buffer'''
        # EN MILISEGUNDOS
        tasa_de_actualizacion = 50
        self.timer = QTimer(self)
        
        # SE TIENE QUE METER EN UNA FUNCION LAMBDA
        # PORQUE LA FUNCION QUE SE ASOCIA A CONNECT
        # NO PUEDE TENER ARGUMENTOS
        self.timer.timeout.connect(lambda: 
                    self.mostrar_imagen(buffer_de_imgs, 
                                        self.canvas))
        
        # CADA 'tasa_de_actualizacion' MSEGUNDOS
        # EL TIMER SE REINICIARÁ Y EJECUTARÁ LA 
        # SEÑAL QUE TIENE CONECTADA ('mostrar_imagen')
        self.timer.start(tasa_de_actualizacion)
        # LANZA EL HILO PARA OBTENER LAS IMAGENES DE LA CAMARA
        self.hilo_de_captura = threading.Thread(target=vc.capturar, 
                                                args=(buffer_de_imgs, ))
        self.hilo_de_captura.start()

    def mostrar_imagen(self, buffer_de_imgs, canvas):
        if not buffer_de_imgs.empty():
            imagen = buffer_de_imgs.get()
            if imagen is not None and len(imagen) > 0:
                img = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                disp_size = img.shape[1], img.shape[0]
                disp_bpl = disp_size[0] * 3
                qimg = QImage(img.data, disp_size[0], disp_size[1], 
                              disp_bpl, QImage.Format_RGB888)
                canvas.dibujar(qimg)

    # ESTAS SON LAS ACCIONES DEL MENU
    # AQUI LAS LLENAS
    def curso(self):
        print("Seleccionaste curso")

    def tema(self):
        print("Seleccionaste tema")
    
    def video(self):
        print("Seleccionaste video")

    # SE EJECUTA AL CERRAR LA VENTANA
    def closeEvent(self, event):
        # MATAMOS EL HILO CAMBIANDO EL VALOR DE
        # CAPTURANDO
        vc.capturando = False
        self.hilo_de_captura.join()
