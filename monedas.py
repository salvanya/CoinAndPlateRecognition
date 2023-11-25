import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from funciones import imshow, imreconstruct, imclearborder

# a) -----------------------------------------------------------------------------------------
# Cargamos una imagen en formato BGR, la convertimos a RGB y la mostramos.
monedas_path = './Media/monedas.jpg'
img = cv2.imread(monedas_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Paso 1: Conversión a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Mostrar imagen
imshow(img, title='Imagen Original', color_img=True)

# Filtrado Gaussiano para suavizar la imagen
blurred = cv2.GaussianBlur(gray, (15, 15), 0)

imshow(blurred, title = "Imagen en gris desenfocada")

# Detección de bordes Canny
edges = cv2.Canny(blurred, 5, 25)

imshow(edges, title = "Detección de bordes Canny")

# Dilatación para cerrar contornos en Canny
dilated_edges = cv2.dilate(edges, None, iterations=2)

imshow(dilated_edges, title = "Bordes dilatados")

# Encontrar contornos en la imagen dilatada
contours, _ = cv2.findContours(dilated_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crear una imagen en blanco para el relleno
fill_img = np.zeros_like(gray)

# Rellenar los contornos con color blanco
cv2.fillPoly(fill_img, contours, color=(255, 255, 255))

#  Dilatación para cerrar contornos en Canny
dilated_edges = cv2.dilate(fill_img, None, iterations=2)

# Encontrar contornos en la imagen dilatada
contours, _ = cv2.findContours(dilated_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Crear una imagen en blanco para el relleno
# fill_img = np.zeros_like(gray)

# # Rellenar los contornos con color blanco
# cv2.fillPoly(fill_img, contours, color=(255, 255, 255))

# Visualizar la imagen de relleno
imshow(fill_img, title='Contornos Rellenos', color_img=False)

# Definir el kernel elíptico para las operaciones morfológicas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))

# Aplicar operación de apertura
opened_img = cv2.morphologyEx(fill_img, cv2.MORPH_OPEN, kernel)

# Visualizar los resultados después de aplicar apertura y clausura
imshow(opened_img, title='Después de Apertura')

# Aplicar la máscara opened_img a la imagen original
final_result = cv2.bitwise_and(img, img, mask=opened_img)

# Visualizar el resultado final
imshow(final_result, title='Imagen Final Segmentada', color_img=True)

# b) ------------------------------------------------------------------------
# Convertir la imagen a escala de grises
gray_segmented = cv2.cvtColor(final_result , cv2.COLOR_RGB2GRAY)

# Umbralizar la imagen para obtener una máscara binaria
_, binary_mask = cv2.threshold(gray_segmented, 1, 255, cv2.THRESH_BINARY)

# Encontrar contornos en la máscara binaria
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Clasificar contornos en monedas y dados
coins = []
dice = []

for contour in contours:
    # Calcular el área y el perímetro del contorno
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calcular la circularidad (4 * pi * área / (perímetro ^ 2))
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # Si la circularidad es alta, clasificar como moneda, de lo contrario como dado
    if circularity > 0.8:
        coins.append(contour)
    else:
        dice.append(contour)

# Crear máscaras para monedas y dados
coins_mask = np.zeros_like(binary_mask)
cv2.drawContours(coins_mask, coins, -1, (255), thickness=cv2.FILLED)

dice_mask = np.zeros_like(binary_mask)
cv2.drawContours(dice_mask, dice, -1, (255), thickness=cv2.FILLED)

# Visualizar las máscaras
plt.figure(figsize=(12, 6))

plt.subplot(121), plt.imshow(coins_mask, cmap='gray'), plt.title('Máscara de Monedas')
plt.subplot(122), plt.imshow(dice_mask, cmap='gray'), plt.title('Máscara de Dados')

plt.show()

# Aplicar la máscara opened_img a la imagen original
coins = cv2.bitwise_and(img, img, mask=coins_mask)
dice = cv2.bitwise_and(img, img, mask=dice_mask)

# Visualizar el resultado final
imshow(coins, title='Imagen Final Segmentada', color_img=True)
# Visualizar el resultado final
imshow(dice, title='Imagen Final Segmentada', color_img=True)

# Convertir la imagen a escala de grises
gray_segmented = cv2.cvtColor(coins, cv2.COLOR_RGB2GRAY)

# Aplicar umbral adaptativo
_, binary_mask = cv2.threshold(gray_segmented, 1, 255, cv2.THRESH_BINARY)

# Encontrar contornos en la máscara binaria
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crear listas para almacenar contornos de cada tipo de moneda
small_coins = []
medium_coins = []
large_coins = []

# Clasificar contornos en monedas por tamaño
for contour in contours:
    # Calcular el área del contorno
    area = cv2.contourArea(contour)
    # Clasificar por tamaño
    if area < 80000:
        small_coins.append(contour)
    elif 80000 <= area <= 100000:
        medium_coins.append(contour)
    elif area > 100000:
        large_coins.append(contour)

# Crear máscaras para cada tipo de moneda
small_coins_mask = np.zeros_like(binary_mask)
cv2.drawContours(small_coins_mask, small_coins, -1, (255), thickness=cv2.FILLED)

medium_coins_mask = np.zeros_like(binary_mask)
cv2.drawContours(medium_coins_mask, medium_coins, -1, (255), thickness=cv2.FILLED)

large_coins_mask = np.zeros_like(binary_mask)
cv2.drawContours(large_coins_mask, large_coins, -1, (255), thickness=cv2.FILLED)

# Visualizar las máscaras
plt.figure(figsize=(12, 6))

plt.subplot(131), plt.imshow(small_coins_mask, cmap='gray'), plt.title('Monedas Pequeñas')
plt.subplot(132), plt.imshow(medium_coins_mask, cmap='gray'), plt.title('Monedas Medianas')
plt.subplot(133), plt.imshow(large_coins_mask, cmap='gray'), plt.title('Monedas Grandes')

plt.show()

# Aplicar la máscara opened_img a la imagen original
small_coins = cv2.bitwise_and(img, img, mask=small_coins_mask)
medium_coins = cv2.bitwise_and(img, img, mask=medium_coins_mask)
large_coins = cv2.bitwise_and(img, img, mask=large_coins_mask)

# Visualizar el resultado final
imshow(small_coins, title='Monedas pequeñas', color_img=True)
# Visualizar el resultado final
imshow(medium_coins, title='Monedas medianas', color_img=True)
# Visualizar el resultado final
imshow(large_coins, title='Monedas grandes', color_img=True)

# Función para contar componentes conexas en una máscara
def contar_componentes_conexas(mask):
    _, labeled_mask = cv2.connectedComponents(mask)
    return labeled_mask

# Contar componentes conexas en cada máscara
labeled_small_coins = contar_componentes_conexas(small_coins_mask)
labeled_medium_coins = contar_componentes_conexas(medium_coins_mask)
labeled_large_coins = contar_componentes_conexas(large_coins_mask)

# Contar el número de componentes en cada máscara
num_small_coins = len(np.unique(labeled_small_coins)) - 1  # Restar 1 para excluir el fondo
num_medium_coins = len(np.unique(labeled_medium_coins)) - 1
num_large_coins = len(np.unique(labeled_large_coins)) - 1

# Mostrar los resultados
print(f"Número de monedas pequeñas: {num_small_coins}")
print(f"Número de monedas medianas: {num_medium_coins}")
print(f"Número de monedas grandes: {num_large_coins}")

# c)------------------------------------------------------------------------------------------
# Obtener los contornos de la imagen segmentada de los dados
contours, _ = cv2.findContours(dice_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Lista para almacenar los dados recortados
dados_recortados = []

# Iterar sobre los contornos de los dados y recortar cada dado
for i, contour in enumerate(contours):
    # Obtener las coordenadas del rectángulo delimitador del contorno
    x, y, w, h = cv2.boundingRect(contour)

    # Recortar el dado de la imagen original
    dice_crop = img[y:y+h, x:x+w]

    # Almacenar el dado recortado en la lista
    dados_recortados.append(dice_crop)

    # Mostrar el dado recortado
    imshow(dice_crop, title=f'Dado {i+1}')


# Lista para almacenar los dados umbralados
dados_umbralados = []

# Iterar sobre los dados recortados y aplicar umbralado invertido
for i, dado_crop in enumerate(dados_recortados):
    # Convertir a escala de grises
    dado_gray = cv2.cvtColor(dado_crop, cv2.COLOR_RGB2GRAY)

    # Aplicar umbral invertido
    _, thresh_dado = cv2.threshold(dado_gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Almacenar el dado umbralado en la lista
    dados_umbralados.append(thresh_dado)

    # Mostrar el dado umbralado
    imshow(thresh_dado, title=f'Dado Umbralado {i+1}')

# Lista para almacenar los dados después de eliminar los elementos que tocan el borde
dados_limpios = []

# Iterar sobre los dados umbralados y eliminar elementos que tocan el borde
for i, dado_umbralado in enumerate(dados_umbralados):
    # Eliminar elementos que tocan el borde
    dado_limpio = imclearborder(dado_umbralado)

    # Almacenar el dado limpio en la lista
    dados_limpios.append(dado_limpio)

    # Mostrar el dado después de eliminar elementos que tocan el borde
    imshow(dado_limpio, title=f'Dado Limpio {i+1}')

# Lista para almacenar los dados después de la operación de apertura
dados_sin_ruido = []

# Definir el kernel elíptico para la operación de apertura
kernel_eliptico = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

# Iterar sobre los dados limpios y aplicar la operación de apertura
for i, dado_limpio in enumerate(dados_limpios):
    # Aplicar la operación de apertura
    dado_sin_ruido = cv2.morphologyEx(dado_limpio, cv2.MORPH_OPEN, kernel_eliptico)

    # Almacenar el dado después de la operación de apertura en la lista
    dados_sin_ruido.append(dado_sin_ruido)

    # Mostrar el dado después de la operación de apertura
    imshow(dado_sin_ruido, title=f'Dado Sin Ruido {i+1}')

# Lista para almacenar los dados después de eliminar objetos elípticos
dados_finales = []

# Umbral para considerar un objeto como elíptico o redondo
umbral_elipticidad = 0.5  

# Iterar sobre los dados sin ruido y eliminar objetos redondos
for i, dado_sin_ruido in enumerate(dados_sin_ruido):
    # Encontrar contornos en el dado sin ruido
    contours, _ = cv2.findContours(dado_sin_ruido, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Eliminar objetos elípticos y conservar los redondo
    dado_final = dado_sin_ruido.copy()
    for contour in contours:
        # Calcular el ajuste de elipticidad
        elipticidad = cv2.fitEllipse(contour)[1][0] / cv2.fitEllipse(contour)[1][1]

        # Eliminar objeto si es redondo
        if elipticidad < umbral_elipticidad:
            cv2.drawContours(dado_final, [contour], -1, 0, thickness=cv2.FILLED)

    # Almacenar el dado después de eliminar objetos redondos en la lista
    dados_finales.append(dado_final)

    # Mostrar el dado después de eliminar objetos redondos
    imshow(dado_final, title=f'Dado Final {i+1}')

# Función para contar componentes conexas en una imagen binaria
def contar_componentes_conexas(img_binaria):
    _, labeled_img = cv2.connectedComponents(img_binaria)
    return labeled_img

# Lista para almacenar el número de componentes conexas para cada dado final
num_componentes_por_dado = []

# Iterar sobre los dados finales y contar componentes conexas
for i, dado_final in enumerate(dados_finales):
    # Contar componentes conexas en el dado final
    labeled_dado = contar_componentes_conexas(dado_final)

    # Obtener el número de componentes (restar 1 para excluir el fondo)
    num_componentes = len(np.unique(labeled_dado)) - 1

    # Almacenar el número de componentes para este dado en la lista
    num_componentes_por_dado.append(num_componentes)

    # Mostrar el dado final con el número de componentes
    imshow(labeled_dado, title=f'Dado Final {i+1}: Muestra el número {num_componentes}', blocking=True)

