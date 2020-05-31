import cv2, glob, random, math, numpy as np, dlib
from sklearn.svm import SVC


emotions = ["aburrido", "confundido","frustrado","concentrado", "neutral"] #Lista de sentimientos
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Archivo de marcas faciales
clf = SVC (kernel='linear', probability=True, tol=1e-3, verbose = True) #Establece el clasificador como un vector de apoyo de máquinas polinomial kernel

def get_files(emotion):
    files = glob.glob("dataset//%s//*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files))] 
    prediction=['dataset//2.jpeg'] #Aqui se añade la imagen que quieres procesar (//para directorio MAC) (\\para Windows)
    return training, prediction

def get_landmarks(image):
   detections = detector(image, 1)
   for k,d in enumerate(detections): #Para todas las cara detectadas de forma individual
      shape = predictor(image, d) 
      xlist = []
      ylist = []
      for i in range(1,68): #Guarda coordenadas X e Y en dos listas
          xlist.append(float(shape.part(i).x))
          ylist.append(float(shape.part(i).y))
 
      xmean = np.mean(xlist) #Obtiene la media de ambos ejes para determinar el centro de gravedad
      ymean = np.mean(ylist)
      xcentral = [(x-xmean) for x in xlist] #Calcula distancia entre cada punto y el punto central en ambos ejes
      ycentral = [(y-ymean) for y in ylist]

      if xlist[26] == xlist[29]: #Si la coordenada x del conjunto son las mismas, el ángulo es 0,  evitamos el error 'divide by 0' en la función
          anglenose = 0
      else:
          anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)

      if anglenose < 0:
          anglenose += 90
      else:
          anglenose -= 90

      landmarks_vectorised = []
      for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
          landmarks_vectorised.append(x)
          landmarks_vectorised.append(y)
          meannp = np.asarray((ymean,xmean))
          coornp = np.asarray((z,w))
          dist = np.linalg.norm(coornp-meannp)
          anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
          landmarks_vectorised.append(dist)
          landmarks_vectorised.append(anglerelative)

   if len(detections) < 1: 
       landmarks_vectorised = "error"
   return landmarks_vectorised

def make_sets():
  training_data = []
  training_labels = []
  prediction_data = []
  prediction_labels = []
  training = []
  prediction = []
  for emotion in emotions:
      training, prediction = get_files(emotion)
      
      for item in training:
          image = cv2.imread(item) #Abrir imagen
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convertimos a escala de grises
          clahe_image = clahe.apply(gray)
          landmarks_vectorised = get_landmarks(clahe_image)
          if landmarks_vectorised == "error":
              pass
          else:
             training_data.append(landmarks_vectorised) #Vector de imágenes a la lista de datos de entrenamiento
             training_labels.append(emotions.index(emotion))
 
      for item in prediction:
          image = cv2.imread(item)
          print(image)
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          clahe_image = clahe.apply(gray)
          landmarks_vectorised = get_landmarks(clahe_image)
          if landmarks_vectorised == "error":
              pass
          else:
             prediction_data.append(landmarks_vectorised)
             prediction_labels.append(emotions.index(emotion))

  return training_data, training_labels, prediction_data, prediction_labels,


probam1 = np.zeros((len(emotions),10))
probam2 = np.zeros((1,len(emotions)))

accur_lin = []

for i in range(0,10):
  print("Making sets %s" %i) #Hace un muestreo aleatorio 80/20%
  training_data, training_labels, prediction_data, prediction_labels = make_sets()

  npar_train = np.array(training_data) #gira el conjunto de entrenamiento en una matriz numpy para el clasificador
  npar_trainlabs = np.array(training_labels)
  print("training SVM linear %s" %i) #Entrenamiento SVM
  clf.fit(npar_train, training_labels)

  print("getting accuracies %s" %i) #Utilice la función score () para obtener mayor precisión
  npar_pred = np.array(prediction_data)
  pred_lin = clf.score(npar_pred, prediction_labels)
  print ("linear: ", pred_lin)
  accur_lin.append(pred_lin) #Guarda la precision en una lista
  proba=clf.predict_proba(prediction_data)
  print ("proba: ", proba)
  probam1[:,i]=proba[1,:]
  probam2=proba[1,:]+probam2
  #probam(:,i)=probam+proba

 
proba=probam2/10 
p1=round(proba[0,0],2)
p2=round(proba[0,1],2)
p3=round(proba[0,2],2)
p4=round(proba[0,3],2)
p5=round(proba[0,4],2)
print("Mean value lin svm: %.3f" %np.mean(accur_lin)) #hacemos 10 ejecuciones para aumentar precision

frame=cv2.imread('dataset//9.jpeg') #aqui se añade la imagen que quieres procesar pero aqui solo se carga para el resultado final
#ploteamos el resultado
cv2.putText(frame, "aburrido: {}".format(p1), (10, 30),
 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(frame, "confundido: {:.2f}".format(p2), (10, 60),
 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(frame, "frustrado: {}".format(p3), (10, 90),
 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(frame, "concentrado: {:.2f}".format(p4), (10, 120),
 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(frame, "neutral: {:.2f}".format(p5), (10, 150),
 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 

 # mostramos la imagen
cv2.imshow("Frame", frame)
cv2.imwrite('resultado.png',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()