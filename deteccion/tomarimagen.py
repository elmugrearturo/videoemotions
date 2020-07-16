"""
	Tomar foto con Python y opencv
"""

import cv2
import uuid


"""
	En este caso, 0 quiere decir que queremos acceder
	a la cámara 0. Si hay más cámaras, puedes ir probando
	con 1, 2, 3...
"""
cap = cv2.VideoCapture(0)

leido, frame = cap.read()

if leido == True:
	nombre_foto = str(uuid.uuid4()) + ".png" # uuid4 regresa un objeto, no una cadena. Por eso lo convertimos
	cv2.imwrite(nombre_foto, frame)
	print("Foto tomada correctamente con el nombre {}".format(nombre_foto))
else:
	print("Error al acceder a la cámara")

"""
	Finalmente liberamos o soltamos la cámara
"""
cap.release()