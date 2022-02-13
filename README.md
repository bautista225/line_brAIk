# Line BrAIk | Estimación de la velocidad de lectura con Keras y OpenCV

## Autores
* Daniel Bagán Martinez
* Aarón García Pitarch
* Ketevan Javakhishvili
* Juan Bautista García Traver

## Objetivo

Con este proyecto se pretende crear una herramienta de soporte al estudio o a la lectura, que ayude a estimar cuáles son las horas del día y los periodos en que el estudiante rinde más, así como la longitud de los intervalos de tiempo en que no baja su concentración, o los momentos en los que el estudiante/lector no está concentrado o está bajando su productividad y debería tomar alguna medida o tomarse un periodo de descanso.

En concreto, la parte del proyecto que se presenta en este repositorio contine el proceso para crear el **modelo encargado de estimar la productividad** del estudiante, usando como magnitud la **tasa de líneas leídas por unidad de tiempo**.

![Captura de pantalla 2022-02-13 a las 20 32 31](https://user-images.githubusercontent.com/25453699/153773790-7169b611-b279-4734-a307-ae9974a81cc6.png)

## Partes del proyecto

### Preprocesamiento de datos

Módulo procesar_video_ojo.py

A partir de un conjunto de videos, crea una carpeta para cada video, una subcarpeta para cada segundo y almacena las 10 imágenes (frames) del ojo.

```python
# Datos de entrada.
carpeta_video = f"videos{os.path.sep}"                # Ruta con los vídeos a procesar

videos_a_analizar = [
   (f"{carpeta_video}v_persona_01.mp4",   "I", True), # "I" indica que debe procesar el ojo izquierdo.
   (f"{carpeta_video}v_persona_02.mp4",   "I", True), # True indica forzar a que siempre se detecten dos ojos,
   (f"{carpeta_video}v_persona_03.mp4",   "I", True), # en otro caso, cogerá el ojo del frame anterior.
   (f"{carpeta_video}v_persona_04.mp4",   "D", True), # "D" indica que debe procesar el ojo derecho.
]
```

### Generación de Dataset

Módulo crear_conjuntos_entrenamiento.ipynb

A partir de las imágenes creadas, almacena el conjunto de datos y las etiquetas en ficheros numpy array .npy

```python
# Datos de entrada

# Indicamos la carpeta de la que obtener los vídeos.
parent_folder = "/Users/equipo-ojo/Documents/OJO/video_inputs/*"

# Las subcarpetas deberán contener un fichero .csv con el nombre del vídeo
# (v_persona_01.csv) donde se encuentren el valor de las etiquetas "y" para cada
# conjunto de 10 frames (1 si hay salto de línea o 0 si no lo hay).

```

### Entrenamiento del modelo

Módulo modelo_line_braik.ipynb

Carga el dataset, crea y entrena el modelo, y muestra resultados.

```python

# Datos de entrada

# Cargamos el conjunto de datos de entrenamiento desde un fichero npy.
x_set_r = load('/path/Saturdays AI - Equipo ojo/dataset/x_set_right.npy') # Dataset del ojo derecho.
y_set_r = load('/path/Saturdays AI - Equipo ojo/dataset/y_set_right.npy') # Etiquetas del ojo derecho.
x_set_l = load('/path/Saturdays AI - Equipo ojo/dataset/x_set_left.npy')  # Dataset del ojo izquierdo.
y_set_l = load('/path/Saturdays AI - Equipo ojo/dataset/y_set_left.npy')  # Etiquetas del ojo izquierdo.

```

### Demo detección de saltos de línea

Módulo demo_line_braik.ipynb

A partir de un video, detecta los saltos de línea y estima la velocidad de lectura.

```python

# Datos de entrada

# Fecha de guardado del vídeo de salida.
now = datetime.now()

video_path = "/path/Saturdays AI - Equipo ojo/Videos/v_persona_01.mp4"  # Ruta al vídeo a procesar.
ai_model_path = '/path/Saturdays AI - Equipo ojo/modeloEntrenado'       # Ruta al modelo entrenado.
video_output_path = f'/path/Saturdays AI - Equipo ojo/video_results/video_output-{now.strftime("%d-%m-%Y_%H-%M-%S")}.mp4'

```
