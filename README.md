# Line BrAIk | Creación de un modelo para la estimación de la velocidad de lectura

## Autores
* Daniel Bagán Martinez
* Aarón García Pitarch
* Ketevan Javakhishvili
* Juan Bautista García Traver

## Objetivo

Con este proyecto se pretende crear una herramienta de soporte al estudio o a la lectura, que ayude a estimar cuáles son las horas del día y los periodos en que el estudiante rinde más, así como la longitud de los intervalos de tiempo en que no baja su concentración, o los momentos en los que el estudiante/lector no está concentrado o está bajando su productividad y debería tomar alguna medida o tomarse un periodo de descanso.

En concreto, la parte del proyecto que se presenta en este repositorio contine el proceso para crear el **modelo encargado de estimar la productividad** del estudiante, usando como magnitud la **tasa de líneas leídas por unidad de tiempo**.

![1_yPQVddKPoXzGLWD3e7PqFA](https://user-images.githubusercontent.com/25453699/154657495-a4b3e848-bea8-4d7a-92a2-f7fb540823ea.gif)

## Partes del proyecto

### Preprocesamiento de datos

Módulo `procesar_video_ojo.py`

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

Módulo `crear_conjuntos_entrenamiento.ipynb`

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

Debido a las limitaciones de hardware en Tensorflow para ejecutar modelos con el formato de entrada NHWC (n_samples, height, width, channels) solamente en entornos GPU Nvidia CUDA, se ha desarrollado la generación de los modelos según su formato de entrada.

#### Entorno con GPU

Módulo `train_model_line_braik_channels_last.ipynb`

> Éste modelo necesitará su ejecución en un entorno GPU debido a la arquitectura de entrada utilizada: 
> NHWC (n_samples, height, width, channels) la cual sólo es compatible con entornos GPU Nvidia CUDA.
> Para utilizar el modelo mediante CPU, utilizar el módulo `train_model_line_braik_channels_first.ipynb`
> que utiliza la arquitectura NCHW (n_samples, channels, height, width).

Carga el dataset, crea y entrena el modelo, y muestra resultados.

```python

# Datos de entrada

# Cargamos el conjunto de datos de entrenamiento desde un fichero npy.
x_set_r = load('/path/Saturdays AI - Equipo ojo/dataset/x_set_right.npy') # Dataset del ojo derecho.
y_set_r = load('/path/Saturdays AI - Equipo ojo/dataset/y_set_right.npy') # Etiquetas del ojo derecho.
x_set_l = load('/path/Saturdays AI - Equipo ojo/dataset/x_set_left.npy')  # Dataset del ojo izquierdo.
y_set_l = load('/path/Saturdays AI - Equipo ojo/dataset/y_set_left.npy')  # Etiquetas del ojo izquierdo.

```

#### Entorno sin GPU

Módulo `train_model_line_braik_channels_first.ipynb`

> Este entorno produce una modificación en el conjunto de entrada
> para pasar el input de la arquitectura NHWC (n_samples, height, width, channels)
> a la arquitectura NCHW (n_samples, channels, height, width)
> ya que la primera posee limitaciones de hardware (sólo compatible con entornos GPU Nvidia CUDA).

Carga el dataset, crea y entrena el modelo, y muestra resultados.

```python

# Datos de entrada

# Cargamos el conjunto de datos de entrenamiento desde un fichero npy.
x_set_r = load('/path/Saturdays AI - Equipo ojo/dataset/x_set_right.npy') # Dataset del ojo derecho.
y_set_r = load('/path/Saturdays AI - Equipo ojo/dataset/y_set_right.npy') # Etiquetas del ojo derecho.
x_set_l = load('/path/Saturdays AI - Equipo ojo/dataset/x_set_left.npy')  # Dataset del ojo izquierdo.
y_set_l = load('/path/Saturdays AI - Equipo ojo/dataset/y_set_left.npy')  # Etiquetas del ojo izquierdo.

```

### Demo detección de saltos de línea con vídeo local

Módulo `demo_line_braik_localvideo.ipynb`

A partir de un video, detecta los saltos de línea y estima la velocidad de lectura.

```python

# Datos de entrada

# Fecha de guardado del vídeo de salida.
now = datetime.now()

video_path = "/path/Saturdays AI - Equipo ojo/Videos/v_persona_01.mp4"  # Ruta al vídeo a procesar.
ai_model_path = '/path/Saturdays AI - Equipo ojo/modeloEntrenado'       # Ruta al modelo entrenado.
video_output_path = f'/path/Saturdays AI - Equipo ojo/video_results/video_output-{now.strftime("%d-%m-%Y_%H-%M-%S")}.mp4'

```

### Demo detección de saltos de línea en tiempo real con webcam

Módulo `demo_line_braik_realtime.py`

A partir de la webcam, detecta los saltos de línea y estima la velocidad de lectura en tiempo real.

```python

# Datos de entrada
ai_model_path = '/path/Saturdays AI - Equipo ojo/modeloEntrenado'       # Ruta al modelo entrenado.

```

### Descarga de modelos entrenados

Modelos entrenados en formato channel_first (input shape 10,80,80) y channel_last (input shape 80,80,10):

[Enlace a Google Drive para la descarga de modelos](https://drive.google.com/drive/folders/1P7QrJeB2OshOMQR-j8c7Xke0S11CuLtb?usp=sharing)

Debido a limitaciones de hardware en Tensorflow, el modelo channel_first solo funciona en entornos con GPU nvidia CUDA, mientras que el modelo channel_last funciona en cualquier entorno.
