# -*- coding: utf-8 -*-
import cv2
import time
import face_recognition

from emotion_detection import detectar_emocion 

# MIENTRAS ESTO SEA VERDADERO
# NO MORIRÁ EL HILO
capturando = True

# OBTENDRA EL ROSTRO Y LO PROCESARÁ
reconociendo = False

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
            if reconociendo:
                # BIBLIOTECA PYTHON ENTRENADA CON DEEP LEARNING
                rostros = face_recognition.face_locations(img)
                # OJO: Algunos pueden no ser caras
                for (top, right, bottom, left) in rostros:
                    img_rostro = img[top:bottom, left:right]
                    emocion = detectar_emocion(img_rostro)
                    img = cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                    img = cv2.putText(img, 
                                      emocion, 
                                      (left, top),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      1,
                                      (0,0,255),
                                      2)

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
