import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure(figsize=(12,6))
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
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