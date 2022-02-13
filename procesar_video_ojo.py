# coding=utf-8
import os
import cv2

def dame_fpsyduracion(video):
    """Devuelve los FPS del vídeo y su duración."""
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    seconds = int(frames / fps)
    return fps,seconds

def detecta_ojo(frame, face_cascade, eye_cascade, ojo_a_analizar, restringir_detectar_dos_ojos):
    """Devuelve la imagen del ojo derecho encontrado, en otro caso, devuelve el frame."""

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    if (len(faces)>0):
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        face_img = frame[y:y+w, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_img, 1.3, 5)

        if (len(eyes) < 2 and restringir_detectar_dos_ojos): # Si no detecta los 2 ojos, que coja el frame anterior.
            return (False, frame)

        if (len(eyes)>0):
            
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

def creaCarpetasSegundo(nombreCarpeta, duracion):
    """Crea una carpeta para guardar los frames y una subcarpeta (p.ej. segundo0042) dentro por cada segundo del video."""

    try:
        os.mkdir(nombreCarpeta)
        for periodo in range(duracion):
            os.mkdir(f"{nombreCarpeta}{os.path.sep}segundo{str(periodo).zfill(4)}")
    except OSError as error:
        print(error)

def framesADescartar(fps, num_frames_borrar):
    """Devuelve un listado de frames a descartar del video."""
    
    framesDescartados = set()

    if num_frames_borrar!=0:
        periodo = round(fps) / num_frames_borrar

        for i in range(num_frames_borrar):
            frame_descartado=round(i*periodo)
            framesDescartados.add(frame_descartado)

    return framesDescartados

def analiza_video(rutaVideo, ojo_a_analizar, restringir_detectar_dos_ojos):
    """Analiza el vídeo de entrada extrayendo una imagen del ojo de los frames deseados."""

    video = cv2.VideoCapture(rutaVideo)
    fps,duracion=dame_fpsyduracion(video)

    rutaVideo_splitted = rutaVideo.split(f"{os.path.sep}")
    nombreVideo = rutaVideo_splitted[len(rutaVideo_splitted) -1]

    nombreCarpetaInterna = f'frames_{nombreVideo.split(".")[0]}'
    carpeta=f'video_inputs{os.path.sep}{nombreCarpetaInterna}'

    # Sets entrenados.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    creaCarpetasSegundo(carpeta, duracion)

    num_frame_global=0
    num_frame_ordenado=0
    num_frame_por_segundo=0

    frame_ojo_prev = []
    img_ext = "jpg"

    num_frames_borrar=round(fps)-10
    framesDescartados = framesADescartar(fps, num_frames_borrar)

    while video.isOpened():

        num_frame_por_segundo=(num_frame_por_segundo+1)%round(fps)
        num_carpeta=(num_frame_global//round(fps))
        path=f"{carpeta}{os.path.sep}segundo{str(num_carpeta).zfill(4)}{os.path.sep}"
        frame_leido, frame = video.read()

        current_second = str(num_frame_ordenado).zfill(4)

        print(f"Voy por el segundo {current_second} de {duracion} del vídeo {nombreVideo}")

        if not frame_leido:
            break

        if num_frame_por_segundo not in framesDescartados:
            frame_reescalado = cv2.resize(frame, (800, 500), interpolation=cv2.INTER_AREA)
            frame_escala_grises = cv2.cvtColor(frame_reescalado, cv2.COLOR_BGR2GRAY)

            success, frame_ojo = detecta_ojo(frame_escala_grises, face_cascade, eye_cascade, ojo_a_analizar, restringir_detectar_dos_ojos)

            # Si no obtenemos ojo en este frame, recuperamos el anterior.
            if (not success and len(frame_ojo_prev)>0): 
                frame_ojo = frame_ojo_prev

            frame_ojo_reescalado = cv2.resize(frame_ojo, (80, 80), interpolation=cv2.INTER_AREA)

            cv2.imwrite(f"{path}{nombreCarpetaInterna}_{current_second}.{img_ext}", frame_ojo_reescalado)

            num_frame_ordenado+=1
            frame_ojo_prev = frame_ojo
        
        num_frame_global+=1

    video.release()


if __name__ == '__main__' :

    # Datos de entrada.
    carpeta_video=f"videos{os.path.sep}"

    videos_a_analizar = [
        (f"{carpeta_video}v_persona_01.mp4",   "I", True),
        (f"{carpeta_video}v_persona_02.mp4",   "I", True),
        (f"{carpeta_video}v_persona_03.mp4",   "D", True),
        (f"{carpeta_video}v_persona_04.mp4",   "I", True),
        ]

    for i in range(len(videos_a_analizar)):
        rutaVideo, ojo_a_analizar, restringir_detectar_dos_ojos = videos_a_analizar[i]
        analiza_video(rutaVideo, ojo_a_analizar, restringir_detectar_dos_ojos)    