import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from funciones import imshow, imreconstruct, imclearborder

# Ruta de la imagen de un vehículo (puedes cambiar el nombre del archivo según tu conjunto de datos)
patente_path = './Media/img04.png'

# Cargar la imagen
img_vehiculo = cv2.imread(patente_path)
img_vehiculo_rgb = cv2.cvtColor(img_vehiculo, cv2.COLOR_BGR2RGB)

# Mostrar la imagen original
imshow(img_vehiculo_rgb, title='Imagen Original del Vehículo', color_img=True)

# Convertir a escala de grises
gray = cv2.cvtColor(img_vehiculo, cv2.COLOR_BGR2GRAY)

# Ajustar imagen
adjusted_image = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)

# Aplicar umbral para resaltar los caracteres
_, thresh = cv2.threshold(adjusted_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


imshow(adjusted_image, title='Imagen ajustada')
imshow(thresh, title='Imagen umbralada')

# Encontrar contornos en la imagen
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar todos los contornos en una copia de la imagen original
img_all_contours = img_vehiculo.copy()
cv2.drawContours(img_all_contours, contours, -1, (0, 255, 0), 1)

# Mostrar la imagen con todos los contornos
imshow(cv2.cvtColor(img_all_contours, cv2.COLOR_BGR2RGB), title='Todos los Contornos')

# Filtrar contornos por área y relación de aspecto
min_area = 500
max_area = 3000  # Puedes ajustar este valor según tus necesidades
min_aspect_ratio = 1.9  # Puedes ajustar este valor según tus necesidades
max_aspect_ratio = 4  # Puedes ajustar este valor según tus necesidades

filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]


filtered_contours = [
   cnt for cnt in filtered_contours
    if min_aspect_ratio < (cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3]) < max_aspect_ratio
]


# Dibujar contornos en una copia de la imagen original
img_filtered_contours = img_vehiculo.copy()
cv2.drawContours(img_filtered_contours, filtered_contours, -1, (0, 255, 0), 2)

# Mostrar la imagen con contornos filtrados
imshow(cv2.cvtColor(img_filtered_contours, cv2.COLOR_BGR2RGB), title='Contornos Filtrados')

# Calcular el bounding rect de los contornos filtrados
bounding_rectangles = [cv2.boundingRect(cnt) for cnt in filtered_contours]

# Dibujar rectángulos en una copia de la imagen original
img_bounding_rectangles = img_vehiculo.copy()
for rect in bounding_rectangles:
    x, y, w, h = rect
    cv2.rectangle(img_bounding_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Mostrar la imagen con los rectángulos del bounding rect
imshow(cv2.cvtColor(img_bounding_rectangles, cv2.COLOR_BGR2RGB), title='Bounding Rectangles')


regiones = []

# Mostrar y recortar las regiones correspondientes en la imagen original
for i, rect in enumerate(bounding_rectangles):
    x, y, w, h = rect
    region_recortada = img_vehiculo[y:y + h, x:x + w]

    plt.subplot(1, len(bounding_rectangles), i + 1)
    plt.imshow(cv2.cvtColor(region_recortada, cv2.COLOR_BGR2RGB))
    plt.title(f'Región {i + 1}')
    plt.axis('off')

    regiones.append(cv2.cvtColor(region_recortada, cv2.COLOR_BGR2RGB))
plt.show()

# Función para aplicar umbralización a una imagen en formato RGB
def threshold_image(rgb_image):
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # Aplicar umbralización
    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresholded_image

# Imprimir la cantidad de componentes conexas en cada imagen antes de filtrar
for i, region in enumerate(regiones):
    num_connected_components = cv2.connectedComponentsWithStats(threshold_image(region), connectivity=4)[0]

# Filtrar imágenes que contienen exactamente seis componentes conexas
filtered_thresholded_regions = [
    threshold_image(region) for region in regiones
    if cv2.connectedComponentsWithStats(threshold_image(region), connectivity=4)[0] > 2
]

# Mostrar imágenes umbralizadas después del filtrado
for i, thresholded_region in enumerate(filtered_thresholded_regions):
    imshow(thresholded_region, title=f'Región {i + 1} Umbralizada')

plt.show()

# Lista vacía para almacenar las regiones procesadas
regiones_procesadas = []

# Iterar sobre las regiones filtradas y umbralizadas
for region in filtered_thresholded_regions:
  
  # Factor de amplificación para aumentar el tamaño de la región
  amplification_factor = 6

  # Amplificar la región mediante redimensionamiento
  region_amplificada = cv2.resize(region, None, fx=amplification_factor, fy=amplification_factor)

  # Aplicar umbral a la imagen amplificada
  _, region_amplificada_umbralado = cv2.threshold(region_amplificada, 128, 255, cv2.THRESH_BINARY)

  # Definir el tamaño del kernel elíptico para la operación de apertura
  kernel_size = 12
  ellipse_size = 1

  # Crear un kernel elíptico
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

  # Aplicar la operación de apertura con el kernel elíptico a la región umbralizada amplificada
  region_amplificada_umbralado_apertura = cv2.morphologyEx(region_amplificada_umbralado, cv2.MORPH_OPEN, kernel)

  # Mostrar la imagen resultante después de la operación de apertura
  imshow(region_amplificada_umbralado_apertura)

  # Limpiar los bordes de la región umbralizada y amplificada
  bordes_limpios = imclearborder(region_amplificada_umbralado_apertura)

  # Agregar la región procesada a la lista
  regiones_procesadas.append(bordes_limpios)

  # Mostrar la imagen resultante después de limpiar los bordes
  imshow(bordes_limpios, title = 'Región sin bordes')

# Función para contar las componentes conexas en una imagen binaria
def count_connected_components(binary_image):
    num_components = cv2.connectedComponents(binary_image)[0]
    return num_components

# Filtrar imágenes que tienen más de 5 componentes conexas
filtered_images_with_six_components = [
    img for img in regiones_procesadas
    if count_connected_components(img) > 5
]

for img in filtered_images_with_six_components:
    imshow(img, 'Región filtrada')

# Tu imagen con las seis componentes conexas
image = filtered_images_with_six_components[0]

# Encontrar contornos en la imagen umbralizada
contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Crear una imagen en negro del mismo tamaño que la original
filled_image = np.zeros_like(image)

# Filtro de área mínimo y máximo
area_min = 600
area_max = 5000

# Filtrar contornos según el área
# Filtrar contornos según el área y relación de aspecto
filtered_contours = [cnt for cnt in contours if
                     area_min < cv2.contourArea(cnt) < area_max and
                     cv2.boundingRect(cnt)[3] > cv2.boundingRect(cnt)[2]]  # Altura mayor que ancho


# Dibujar contornos filtrados con relleno en la imagen en negro
cv2.drawContours(filled_image, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# Mostrar la imagen resultante
imshow(filled_image, title = 'Caracteres de la patente', blocking = True)