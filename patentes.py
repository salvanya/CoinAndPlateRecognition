from funciones import crearListaNombresArchivo, cargarImagenes, preprocesarImagenes, imprimirPreprocesadas, deteccionPatentes, imprimirPatentesDetectadas, encuadrarCaracteres

# Crear la lista con los nombres de los archivos
lista_nombres_archivo = crearListaNombresArchivo()

# Cargar las imágenes en escalas de gris
imagenes_gris = cargarImagenes(lista_nombres_archivo)

# Preprocesar imágenes
imagenes_preprocesadas = preprocesarImagenes(imagenes_gris)

# Imprimir estadíos del preprocesamiento
imprimirPreprocesadas(imagenes_preprocesadas)

# Detectar patentes
patentes_econtradas = deteccionPatentes(imagenes_preprocesadas)

imprimirPatentesDetectadas(patentes_econtradas)

patente_caracteres_encuadrados = encuadrarCaracteres(patentes_econtradas)