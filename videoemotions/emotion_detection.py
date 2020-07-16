# -*- coding: utf-8 -*-
import cv2
from keras.models import load_model
import numpy as np

modelo = load_model("../keras_emotion_model/emo_mod.hdf5")
emociones= {'Enojo': 0, 
            'Tristeza': 5, 
            'Neutral': 4, 
            'Asco': 1, 
            'Sorpresa': 6, 
            'Miedo': 2, 
            'Alegría': 3}
emociones_lookup = dict((v,k) for k,v in emociones.items())

def detectar_emocion(raw_bgr_face_img):
    '''regresa 1 de 7 emociones'''
    
    rostro = cv2.resize(raw_bgr_face_img, (48,48))
    rostro_gris = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
    rostro_gris = np.reshape(rostro_gris, 
                             [1, rostro_gris.shape[0], rostro_gris.shape[1], 1])
    emocion_predicha = np.argmax(modelo.predict(rostro_gris))
    return emociones_lookup[emocion_predicha]
