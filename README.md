# Line BrAIk

## Estimación de la velocidad de lectura con Keras y OpenCV

Con este proyecto se pretende crear una herramienta de soporte al estudio o a la lectura, que ayude a estimar cuáles son las horas del día y los periodos en que el estudiante rinde más, así como la longitud de los intervalos de tiempo en que no baja su concentración, o los momentos en los que el estudiante/lector no está concentrado o está bajando su productividad y debería tomar alguna medida o tomarse un periodo de descanso.

En concreto, la parte del proyecto que se presenta en este repositorio contine el proceso para crear el **modelo encargado de estimar la productividad** del estudiante, usando como magnitud la **tasa de líneas leídas por unidad de tiempo**.

![Captura de pantalla 2022-02-13 a las 20 32 31](https://user-images.githubusercontent.com/25453699/153773790-7169b611-b279-4734-a307-ae9974a81cc6.png)

## Estructura 

1. **procesar_video_ojo.py**. A partir de un conjunto de videos, crea una carpeta para cada video, una subcarpeta para cada segundo y almacena las 10 imágenes del ojo.

2. **crear_conjuntos_entrenamiento.ipynb**. A partir de las imágenes creadas, almacena el conjunto de datos y las etiquetas en ficheros .npy.

3. **modelo_line_braik.ipynb**. Carga el dataset, crea el modelo, lo entrena y muestra resultados.

4. **demo_line_braik.ipynb**. A partir de un video, detecta los saltos de línea y estima la velocidad de lectura.

**x_set_left.npy**. Dataset del ojo izquierdo.

**x_set_right.npy**. Dataset del ojo derecho.

**y_set_left.npy**. Etiquetas del ojo izquierdo.

**y_set_right.npy**. Etiquetas del ojo derecho.
