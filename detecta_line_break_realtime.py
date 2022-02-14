#
# Importaciones
#

from keras.models import load_model
import numpy as np
import os
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

#
# Definiciones y métodos
#

def detecta_ojo(frame, face_cascade, eye_cascade, ojo_a_analizar="D", restringir_detectar_dos_ojos=True):
    """Devuelve la imagen del ojo derecho encontrado, en otro caso, devuelve el frame."""
    
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        face_img = frame[y:y+w, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_img, 1.3, 5)

        if len(eyes) < 2 and restringir_detectar_dos_ojos: # Si no detecta los 2 ojos, que coja el frame anterior.
            return (False, frame)

        if len(eyes) > 0:
            
            pos_ojo = get_posicion_ojo(ojo_a_analizar, eyes, faces[0])

            ex, ey, ew, eh = eyes[pos_ojo]
            eye_img = face_img[ey:ey+eh,ex:ex+ew]

            return (True, eye_img)

    return (False, frame)

def get_posicion_ojo(tipo_ojo, ojos_detectados, cara):
    """Devuelve la posición del ojo deseado (izquierdo en el vídeo o derecho) en ojos_detectados."""
    
    x, y, w, h = cara
    pos_ojo=0
    x_ojo=0

    isOjoALaDerecha = tipo_ojo == "D"
    isOjoALaIzquierda = tipo_ojo == "I"

    if isOjoALaIzquierda:
        x_ojo=x+w

    for i in range(len(ojos_detectados)):

        if isOjoALaIzquierda and ojos_detectados[i][0] < x_ojo:
            x_ojo=ojos_detectados[i][0]
            pos_ojo=i

        if isOjoALaDerecha and ojos_detectados[i][0] > x_ojo:
            x_ojo=ojos_detectados[i][0]
            pos_ojo=i

    return pos_ojo

def hasBreakLine(modelo_final, frames_ojo_list):
    '''Devuelve true si se trata de una línea leída.'''
    x_test_prueba_set = np.array([frames_ojo_list])
    x_test_prueba_set = np.rollaxis(x_test_prueba_set, 3, 1)#funciona para 80,80,10
    x_test_prueba_set = np.rollaxis(x_test_prueba_set, 3, 1)
    results = np.argmax(modelo_final.predict(x_test_prueba_set), axis = 1)

    return results[0] == 1

def getImageShape(image):
  """Devuelve las dimensiones de la imagen."""
  
  height, width = image.shape[:2]

  return height, width

def changeImageColor(target_img):
  """Devuelve una imagen con un filtro de color aplicado."""
  
  #target_img = cv2.cvtColor(target_img,cv2.COLOR_GRAY2RGB) # Aplicar solo si la imagen de entrada está en escala de grises.
  
  # Creamos dos copias de la imagen:
  # · Una para la capa de encima.
  # · Otra para la salida.
  overlay = target_img.copy()
  output = target_img.copy()

  img_h, img_w = getImageShape(target_img)

  # Dibujamos un rectángulo en la imagen.

  cv2.rectangle(overlay, (0, 0), (img_w, img_h), (0, 255, 0), -1)
  
  # Aplicamos la capa.
  cv2.addWeighted(overlay, 0.4, output, 1 - 0.4, 0, output)

  return output

def addInfoLineasLeidas(target_img, numLineasLeidas, segundoEnCurso):
    """Devuelve una imagen con la info de líneas leidas y segundos del vídeo."""

    # Creamos dos copias de la imagen:
    # · Una para la capa de encima.
    # · Otra para la salida.
    overlay = target_img.copy()
    output = target_img.copy()

    blue=188.955
    green=113.985
    red=0

    font_size=0.8
    font_weight=2

    # Añadimos el texto.
    cv2.putText(overlay, f"{numLineasLeidas} LINEAS LEIDAS EN {segundoEnCurso} SEGUNDOS", 
                (20, 60), cv2.FONT_HERSHEY_COMPLEX, font_size, (blue, green, red), font_weight)

    cv2.putText(overlay, f"TASA: {tasaLeidoSegundo(numLineasLeidas, segundoEnCurso)} LINEAS/SEGUNDO", 
                (20, 100), cv2.FONT_HERSHEY_COMPLEX, font_size, (blue, green, red), font_weight)
    
    alpha=0.8

    # Aplicamos la capa.
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output

def cargaCamara():
  '''Obtiene el vídeo de la cámara y sus dimensiones.'''

  video = cv2.VideoCapture(0)
  seconds = 0

  width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

  return video, width, height

def reescalaFrame(frame):
    """Devuelve el frame reescalado para el trabajo con la red."""

    frame_reescalado = cv2.resize(frame,(800,600),fx=0,fy=0, interpolation = cv2.INTER_AREA)

    return frame_reescalado

def reescalaOjo(frame_ojo):
    '''Devuelve el frame del ojo reescalado para el trabajo con el modelo.'''

    frame_ojo_reescalado = cv2.resize(frame_ojo, (80, 80), interpolation=cv2.INTER_AREA)  

    return frame_ojo_reescalado  

def frameToGrayScale(frame):
    """Devuelve el frame en escala de grises para el trabajo con la red."""

    frame_escala_grises = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame_escala_grises

def tasaLeidoSegundo(numLineasLeidas, segundoEnCurso):
    """Devuelve la tasa de líneas leídas por segundo."""

    tasa = segundoEnCurso
    
    if segundoEnCurso != 0:
        tasa = round(numLineasLeidas/segundoEnCurso, 3)

    return tasa

def getSecondsFromDate(init_date):
    """Devuelve la diferencia entre dos fechas en segundos."""

    future_date = datetime.now()
    past_date = init_date

    difference = (future_date - past_date)

    total_seconds = difference.total_seconds()

    return total_seconds

#
# Ejecución del programa principal
#

if __name__ == "__main__":

    # Parámetros de entrada.
    ai_model_path = '/Users/juanbautista/Documents/OJO/modelo_94shape_buena'
    
    # Cargamos el modelo con el que predecir los saltos de línea.
    modelo_final = load_model(ai_model_path)

    # Cargamos los clasificadores.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Obtenemos el vídeo y metadatos.
    capture, width, height = cargaCamara()

    # Variables de entorno.
    frame_ojo_prev=[]
    frames_ojo_list=[]
    frames_list=[]
    frame_counter = 0

    frames_a_colorear = 0
    num_lineas_leidas = 0

    fecha_inicio = datetime.now()

    while True:

        hasFrame, frame = capture.read()

        if not hasFrame:
            break

        print(f"\rVoy por el frame {frame_counter} en {getSecondsFromDate(fecha_inicio)} segundos.", end="")
        
        frame_counter+=1
        
        frame_reescalado = reescalaFrame(frame)
        frame_escala_grises = frameToGrayScale(frame_reescalado)

        ojo_detectado, frame_ojo = detecta_ojo(frame_escala_grises, face_cascade, eye_cascade)

        # Si no obtenemos ojo en este frame, recuperamos el anterior.
        if not ojo_detectado and len(frame_ojo_prev) > 0: 
            frame_ojo = frame_ojo_prev

        frame_ojo_reescalado = reescalaOjo(frame_ojo)

        frames_ojo_list.append(frame_ojo_reescalado)
        frames_list.append(frame)

        current_frame = frame

        if frames_a_colorear > 0:
            current_frame = changeImageColor(frame)
            frames_a_colorear -= 1
        
        current_seconds = getSecondsFromDate(fecha_inicio)

        readingRate = tasaLeidoSegundo(num_lineas_leidas, current_seconds)

        current_frame = addInfoLineasLeidas(current_frame, num_lineas_leidas, current_seconds)

        cv2.imshow("Deteccion salto linea en tiempo real", current_frame)

        # Acumulamos los 10 frames con los que trabaja el modelo.
        if len(frames_ojo_list) == 10:

            isBreakLine = hasBreakLine(modelo_final, frames_ojo_list)
            
            if isBreakLine:
                frames_a_colorear = 10
                num_lineas_leidas += 1

            for i in range(len(frames_ojo_list)):
                current_frame = frames_list[i]

                # Realizamos las modificaciones sobre el frame.
                if isBreakLine:
                    current_frame = changeImageColor(current_frame)

            # Reseteamos variables de entorno.
            frames_ojo_list = []
            frames_list = []

        frame_ojo_prev = frame_ojo

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Finalizamos ejecución y guardamos el vídeo
    capture.release()
    cv2.destroyAllWindows()