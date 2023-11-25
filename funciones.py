import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    # Crear una nueva figura si new_fig es True
    if new_fig:
        plt.figure(figsize=(12, 6))

    # Mostrar la imagen en color si color_img es True, de lo contrario, en escala de grises
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')

    # Establecer el título de la figura si se proporciona
    plt.title(title)

    # Ocultar los ejes si ticks es False
    if not ticks:
        plt.xticks([]), plt.yticks([])

    # Mostrar la barra de color si colorbar es True
    if colorbar:
        plt.colorbar()

    # Mostrar la figura, bloqueando si blocking es True
    if new_fig:
        if blocking:
            plt.show(block=True)
        else:
            plt.show()

# Definimos una función para la reconstrucción morfológica
def imreconstruct(marker, mask, kernel=None):
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatación
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Intersección
        if (marker == expanded_intersection).all():                         # Comprobamos si hemos terminado
            break
        marker = expanded_intersection
    return expanded_intersection

# Función para eliminar objetos que tocan el borde utilizando reconstrucción morfológica
def imclearborder(img):
    marker = img.copy()
    marker[1:-1, 1:-1] = 0
    border_elements = imreconstruct(marker=marker, mask=img)
    img_cb = cv2.subtract(img, border_elements)
    return img_cb

# Crear lista de nombres de archivos
def crearListaNombresArchivo():
    # Inicializar una lista vacía para almacenar los nombres de archivo
    lista_nombres_archivo = []

    # Iterar desde 1 hasta 12 (inclusive)
    for i in range(1, 13):
        # Formatear el número como cadena y agregar un cero al frente si es menor que 10
        if i < 10:
            numero = '0' + str(i)
        else:
            numero = str(i)

        # Crear el nombre de archivo en el formato 'imgXX.png' y agregarlo a la lista
        lista_nombres_archivo.append(f'img{numero}.png')

    # Devolver la lista de nombres de archivo creada
    return lista_nombres_archivo

# Cargar imágenes desde una lista de nombres de archivo
def cargarImagenes(lista_nombres_archivo, color=False):
    # Inicializar una lista vacía para almacenar las imágenes cargadas
    loaded_imgs = []

    # Iterar sobre los nombres de archivo en la lista
    for nombre_archivo in lista_nombres_archivo:
        # Leer la imagen en formato BGR
        img_bgr = cv2.imread(f'./Media/{nombre_archivo}')

        # Convertir la imagen a escala de grises o a formato RGB según la bandera 'color'
        if color:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Agregar la imagen a la lista de imágenes cargadas
        loaded_imgs.append(img)

    # Devolver la lista de imágenes cargadas
    return loaded_imgs

def imadjust(x, vin=None, vout=[0, 255], gamma=1):
    # x      : Imagen de entrada en escalas de grises (2D), formato uint8.
    # vin    : Límites de los valores de intensidad de la imagen de entrada
    # vout   : Límites de los valores de intensidad de la imagen de salida
    # y      : Imagen de salida

    # Si vin no se proporciona, calculamos automáticamente los límites utilizando los valores mínimo y máximo de x
    if vin == None:
        vin = [x.min(), x.max()]

    # Aplicamos la corrección gamma y mapeamos los valores de intensidad
    y = (((x - vin[0]) / (vin[1] - vin[0])) ** gamma) * (vout[1] - vout[0]) + vout[0]

    # Recortamos valores que están por debajo de los límites de salida vout
    y[x < vin[0]] = vout[0]   # Valores menores que low_in se mapean a low_out
    y[x > vin[1]] = vout[1]   # Valores mayores que high_in se mapean a high_out

    # Si x es de tipo uint8, aseguramos que los valores estén en el rango de 0 a 255
    if x.dtype == np.uint8:
        y = np.uint8(np.clip(np.round(y), 0, 255))   # Numpy underflows/overflows para valores fuera de rango, se debe utilizar clip.

    # Devolvemos la imagen ajustada
    return y

def preprocesarImagenes(lista_imagenes_gris):
  # Inicializar 
  preprocessed_img = []

  # Iterar sobre cada imagen
  for imagen_gris in lista_imagenes_gris:
      # Ajustar el contraste con la función imadjust------------------------------------------
      adjusted_img = imadjust(imagen_gris, vin=[0,180] , vout=[0, 255], gamma=1)

      # Ampliar las imágenes ----------------------------------------------------------------
      # Obtiene las dimensiones originales de la imagen
      height, width = adjusted_img.shape[:2]

      # Define el nuevo tamaño (el doble del tamaño original)
      width_new = width * 2
      height_new = height * 2

      # Realiza el redimensionamiento utilizando cv2.resize
      resized_img = cv2.resize(adjusted_img, (width_new, height_new))

      # Umbralización -------------------------------------------------------------------------
      _, threshold_img = cv2.threshold(resized_img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

      # Top Hat -------------------------------------------------------------------------------
      # Crear un kernel para la tranformación morfológica de Top-Hat
      kernel_tophat = np.ones((15, 15), np.uint8)

      #Aplicar Top-Hat
      tophat_img = cv2.morphologyEx(threshold_img, cv2.MORPH_TOPHAT, kernel_tophat)

      # Erosión -------------------------------------------------------------------------------
      # Crear un elemento estructural en forma de elipse para la erosión
      kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))

      # Aplicar erosión
      eroded_img = cv2.erode(tophat_img, kernel_erode,iterations=1)

      # Guardar imágenes ----------------------------------------------------------------------
      # Crear lista de las etapas del preoceso de preprocesamiento
      process = [imagen_gris, adjusted_img, threshold_img, tophat_img , eroded_img]

      # Añadir los estadíos de preprocesamiento de esta imagen a la lista 
      preprocessed_img.append(process)
  
  # Devolver lista con los estadíos del preprocesamiento de las imágenes
  return preprocessed_img

# Función para medir la distancia euclidiana entre componentes conexas
def distanciaEuclidiana(componenteA, componenteB):
    # Desempaquetar las coordenadas de componenteA y componenteB
    x1, y1 = componenteA[0] + componenteA[2], componenteA[1]
    x2, y2 = componenteB[0], componenteB[1]
    
    # Calcular la distancia euclidiana entre los dos puntos
    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Devolver la distancia calculada
    return distancia

def filtrarComponentesPorArea(imagen, num_labels, labels, stats, centroids):
    # Inicializar las listas para los contornos filtrados
    filtered_labels = []
    filtered_stats = []
    filtered_centroids = []  

    # Intervalo de áreas para filtrar componentes
    area_min = 80
    area_max = 480

    # Inicializo una imagen en negro para dibujar las componentes filtradas
    imagenes_gris_componentes_filtradas = np.zeros_like(imagen)

    # Filtrar por área
    for label in range(1, num_labels): 
        # Obtener el área de la componente etiquetada
        area = stats[label, cv2.CC_STAT_AREA]

        # Verificar si el área está dentro del rango deseado
        if area > area_min and area < area_max:
            # Agregar la etiqueta, estadísticas y centroides a las listas filtradas
            filtered_labels.append(label)
            filtered_stats.append(stats[label])
            filtered_centroids.append(centroids[label])

            # Resaltar la componente en la imagen filtrada (blanco)
            imagenes_gris_componentes_filtradas[labels == label] = 255

    # Devolver la imagen con las componentes filtradas resaltadas
    return (imagenes_gris_componentes_filtradas, filtered_labels, filtered_stats, filtered_centroids)

def filtrarComponentesPorRAspecto(imagen, filtered_labels, filtered_stats, centroids, labels):
    # Intervalo de relaciones de aspecto para filtrar componentes
    min_aspect_ratio = 0.4
    max_aspect_ratio = 0.7

    # Crear una imagen en negro para dibujar las nuevas componentes filtradas
    componentes_filtradas = np.zeros_like(imagen)

    # Filtrar componentes conectadas por relación de aspecto
    filtered_stats_aspect = [
                             stat 
                             for stat in filtered_stats 
                             if min_aspect_ratio < stat[2] / stat[3] < max_aspect_ratio
                            ]

    # Filtrar centroides por relación de aspecto usando las etiquetas ya filtradas
    filtered_centroids_aspect = [
                                 centroids[label]
                                 for label, stat in zip(filtered_labels, filtered_stats)
                                 if any(np.array_equal(stat, s) for s in filtered_stats_aspect)
                                ]

    # Inicializar una lista vacía para almacenar las etiquetas filtradas
    filtered_labels_aspect = []

    # Obtener las etiquetas que cumplen con la condición de aspecto
    filtered_labels_aspect = [
                              label
                              for label, stat in zip(filtered_labels, filtered_stats)
                              if any(np.array_equal(stat, s) for s in filtered_stats_aspect)
                             ]

    # Resaltar las componentes en la imagen filtrada
    componentes_filtradas[np.isin(labels, filtered_labels_aspect)] = 255

    # Devolver la imagen con las componentes filtradas por aspecto resaltadas
    return componentes_filtradas,filtered_centroids_aspect, filtered_labels_aspect, filtered_stats_aspect

def resaltarComponentesCercanas(imagen, filtered_labels_aspect, filtered_stats_aspect, labels):
    # Crear una imagen negra para colocar el nuevo filtrado
    componentes_filtradas = np.zeros_like(imagen)

    # Definir umbral de distancia
    umbral_distancia = 30

    # Lista para almacenar las etiquetas filtradas por distancia
    label_filtrada_distancia = []

    # Iterar sobre las etiquetas y estadísticas filtradas por aspecto
    for label, stat in zip(filtered_labels_aspect, filtered_stats_aspect):
        
        # Iterar nuevamente para comparar con otras etiquetas y estadísticas
        for s_label, s_stat in zip(filtered_labels_aspect, filtered_stats_aspect):

            # Verificar si la etiqueta actual es diferente y la distancia es menor que el umbral
            if label != s_label and distanciaEuclidiana(stat, s_stat) < umbral_distancia:
                
                # Resaltar las componentes en la imagen filtrada
                componentes_filtradas[labels == label] = 255
                componentes_filtradas[labels == s_label] = 255
                
                # Agregar la etiqueta actual a la lista filtrada por distancia
                label_filtrada_distancia.append(label)
                
                # Salir del bucle interno después de encontrar una coincidencia
                break  

    # Devolver la imagen con las componentes filtradas por distancia resaltadas
    return componentes_filtradas

def calcularDistanciaCentroides(num_labels, centroids):
    # Calcular distancias entre centroides
    distancias = np.zeros((num_labels, num_labels), dtype=np.float32)

    for i in range(1, num_labels):
        for j in range(i + 1, num_labels):
            distancias[i, j] = np.linalg.norm(centroids[i] - centroids[j])
            distancias[j, i] = distancias[i, j]
    return distancias

def componentesDeAgrupadas(componentes_filtradas):
    # Definir la distancia máxima para considerar componentes como agrupadas
    distancia_maxima = 200

    # Aplicar la función connectedComponentsWithStats para obtener información sobre las componentes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(componentes_filtradas, 4)

    # Calcular distancias entre centroides
    distancias = calcularDistanciaCentroides(num_labels, centroids)
    
    # Filtrar componentes conectadas
    componentes_filtradas_agrupadas = []

    for i in range(1, num_labels):
        # Encontrar índices de componentes adyacentes basándose en distancias
        componentes_adyacentes = np.where(distancias[i] < distancia_maxima)[0]

        # Incluir el componente actual solo si tiene al menos 5 componentes adyacentes
        if len(componentes_adyacentes) >= 5:
            componentes_filtradas_agrupadas.append(i)

    # Crear una imagen para visualizar las componentes conectadas filtradas
    imagen_comp_conect = np.zeros_like(labels, dtype=np.uint8)

    for componente in componentes_filtradas_agrupadas:
        # Seleccionar solo los píxeles de la componente actual
        mascara_componente = np.uint8(labels == componente)
        imagen_comp_conect += mascara_componente * componente

    return imagen_comp_conect

def recortarImagen(imagen):
    # Aplicar la función connectedComponentsWithStats para obtener información sobre las componentes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen, 4)

    # Obtener las coordenadas y dimensiones del rectángulo que encierra las componentes
    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(np.uint8(labels > 0))

    # Añadir un margen de 15 píxeles alrededor del rectángulo y recortar la imagen original
    imagen_recortada = imagen[y_rect - 15: y_rect + h_rect + 15, x_rect - 15: x_rect + w_rect + 15]

    return imagen_recortada

def deteccionPatentes(lista_imagenes_procesadas):
    # Inicializar lista de estadíos de detección de patentes
    lista_patentes = []

    for imagen in lista_imagenes_procesadas:
        # Quedarse con la imagen corresponidente al último estadío del preprocesamiento
        imagen_contornos = imagen[4].copy()

        # Encuentra componentes las componentes conexas -----------------------------------------------------------
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_contornos, 4)
        
        # Filtrar componentes conexas según su área ----------------------------------------------------------------
        stats_filtradas = filtrarComponentesPorArea(imagen_contornos, num_labels, labels, stats, centroids)
        imagenes_gris_componentes_filtradas, filtered_labels, filtered_stats, filtered_centroids = stats_filtradas

        # Filtrar por relación de aspecto --------------------------------------------------------------------------
        stats_filtradas = filtrarComponentesPorRAspecto(imagen_contornos, filtered_labels, filtered_stats, centroids, labels)
        componentes_filtradas,filtered_centroids_aspect, filtered_labels_aspect, filtered_stats_aspect = stats_filtradas

        # Eliminar componentes aisladas -----------------------------------------------------------------------------------
        componentes_juntas = resaltarComponentesCercanas(imagen_contornos, filtered_labels_aspect, filtered_stats_aspect, labels)
        
        # Buscar componentes agrupadas de a 6 o más --------------------------------------------------------------------
        imagen_comp_conect = componentesDeAgrupadas(componentes_juntas)
        
        # Recortar la imagen resultante --------------------------------------------------------------------------------------
        imagen_recortada= recortarImagen(imagen_comp_conect)

        # Almacentar los estadíos de la detección para esta imagen -----------------------------------------------------------
        # Crear lista de estadíos de detección
        imagenes_patentes = [imagen_contornos, componentes_juntas, imagen_comp_conect, imagen_recortada]

        # Agregar los estadíos a la lista
        lista_patentes.append(imagenes_patentes)

    # Devolver los estadíos de la detección hallados
    return lista_patentes

def encuadrarCaracteres(lista_patentes):
    # Lista para almacenar las imágenes encuadradas
    imagenes_encuadradas = []

    # Iterar sobre cada conjunto de imágenes en la lista de patentes
    for index, imagenes in enumerate(lista_patentes):
        # Seleccionar la imagen de los caracteres
        imagen = imagenes[3]

        # Verificar si la imagen no es nula y no está completamente en negro
        if imagen is not None and not np.all(imagen == 0):
            # Aplicar umbralización a la imagen
            _, imagen_umbralizada = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY)

            # Crear una imagen en color (BGR) para dibujar rectángulos
            imagen_con_rectangulos = cv2.cvtColor(imagen_umbralizada, cv2.COLOR_GRAY2BGR)
        else:
            continue  # Salir de la iteración si la imagen es nula o completamente en negro

        # Encontrar los contornos en la imagen umbralizada
        contornos, jerarquia = cv2.findContours(imagen_umbralizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dibujar rectángulos verdes alrededor de los contornos
        for contorno in contornos:
            x, y, w, h = cv2.boundingRect(contorno)
            cv2.rectangle(imagen_con_rectangulos, (x, y), (x + w, y + h), (0, 255, 0), 1) 

        # Mostrar la imagen original y la imagen con los rectángulos dibujados
        imshow(imagen_con_rectangulos, title=f'Caracteres detectados en la Patente {index + 1}')

        # Añadir la imagen con rectángulos a la lista de imágenes encuadradas
        imagenes_encuadradas.append(imagen_con_rectangulos)

    # Mostrar todas las imágenes encuadradas
    plt.show()

    # Devolver la lista de imágenes encuadradas
    return imagenes_encuadradas

def imprimirPreprocesadas(lista_preprocesadas):
    # Definir títulos para cada tipo de imagen preprocesada
    titulos = {
        0: 'Imagen en escala de grises N° ',
        1: 'Imagen ajustada N° ',
        2: 'Imagen Ubralizada N° ',
        3: 'Imagen con transformación Top-Hat N° ',
        4: 'Imagen erosionada N° '
    }

    # Iterar sobre cada imagen en la lista preprocesada
    for j, imagen in enumerate(lista_preprocesadas):
        # Iterar sobre los tipos de imagen preprocesada
        for i in range(5):
            # Mostrar cada tipo de imagen preprocesada utilizando la función imshow
            imshow(imagen[i], title=titulos[i] + str(j + 1))

def imprimirPatentesDetectadas(lista_patentes):
    # Lista para almacenar las patentes detectadas
    patentes = []

    # Definir títulos para cada tipo de imagen en la lista de patentes
    titulos = {
        0: 'Imagen procesada N° ',
        1: 'Imagen con filtrada N° ',
        2: 'Caracteres de la patente N° ',
        3: 'Patente N° '
    }

    # Iterar sobre cada conjunto de imágenes en la lista de patentes
    for j, imagen in enumerate(lista_patentes):
        # Iterar sobre los tipos de imagen en cada conjunto
        for i in range(4):
            # Verificar si la imagen no está completamente en negro
            if not np.all(imagen[i] == 0):
                # Mostrar cada tipo de imagen en la lista de patentes utilizando la función imshow
                imshow(imagen[i], title=titulos[i] + str(j + 1))

    # Devolver la lista de patentes detectadas
    return patentes

