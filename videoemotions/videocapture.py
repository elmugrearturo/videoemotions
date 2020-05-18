# -*- coding: utf-8 -*-
import cv2
import time

# MIENTRAS ESTO SEA VERDADERO
# NO MORIRÁ EL HILO
capturando = True

# ESTA FUNCIÓN ESTARÁ EN UN HILO EJECUTANDOSE
def capturar(buffer_de_imgs, 
             camara=0, 
             resolucion=(1920, 1080),
             actualizacion_display=50):
    cap = cv2.VideoCapture(camara)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolucion[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucion[1])
    
    # MODIFICANDO TIEMPO DE EXPOSICION (camara opaca)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -1)
    while capturando:
        if cap.grab():
            retval, img = cap.retrieve(0)
            # AQUI YA TIENES LA IMAGEN CAPTURADA!
            # ENTONCES PUEDES MANDARLA (CON O SIN BUFFER)
            # A TU CÓDIGO DE DETECCIÓN
            if img is not None and buffer_de_imgs.qsize() < 2:
                buffer_de_imgs.put(img)
            else:
                # ESTO ESPERA TANTITO PARA QUE ALCANCE
                # A DIBUJAR EN PANTALLA
                time.sleep(actualizacion_display / 1000.0)
        else:
            print("No se pudo abrir la camara")
            break
    cap.release()
