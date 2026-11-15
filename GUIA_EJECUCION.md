# Proyecto universitario: VGG-16 frente a YOLO26n-cls

## Qué está preparado
Comparación de **clasificación de imágenes** en las cuatro etiquetas originales del dataset. Ambos modelos reciben radiografías completas, con el mismo ajuste proporcional a 224×224 y relleno negro. Las cajas sirven para obtener la clase durante la preparación; no se usan como entrada ni para recortar las imágenes en inferencia. Se preserva la partición agrupada ya depurada.

Modelo base: VGG-16 con transferencia de aprendizaje, apoyado en el artículo de Baymurat et al. (2024), detallado en `informe/seleccion_y_referencias.md`. Es una **adaptación**, ya que cambian datos, clases, entorno y algunos parámetros. El paquete no acredita una reproducción exacta del resultado original.

Modelo de comparación: YOLO26n-cls, variante de clasificación publicada en la documentación oficial consultada el 14/11/2026. No confundir con YOLO detección ni con el formato de almacenamiento YOLOv5 del ZIP.

## Cómo empezar en Colab
1. Abrir [el notebook publicado](https://colab.research.google.com/github/Sneyderx/comparacion-escoliosis-vgg16-yolo/blob/main/Colab_escoliosis.ipynb).
2. Activar una GPU en el entorno de ejecución.
3. Subir `scoliosis_depurado.zip` a la raíz de Mi unidad de Drive.
4. Ejecutar las celdas en orden: la primera clona el repositorio automáticamente.
5. Ejecutar la prueba corta antes de un experimento nuevo. Sus métricas no son resultados experimentales.
6. Ejecutar entrenamiento, evaluación y resumen. Los resultados se guardan en Drive.

## Protocolo
Ver `config.json`. Cinco semillas emparejadas: 11, 22, 33, 44 y 55. Máximo 50 épocas, paciencia 10, selección por F1 macro de validación con desempate por pérdida. SGD, tasa 0.0003, momentum 0.9, L2 0.004, lote 16 y dropout 0.5. Se ajustan todas las capas de ambos modelos desde pesos ImageNet. La misma rotación ±8° y probabilidad de reflejo 0.5 se aplican solo a entrenamiento. Para cada época/imagen/semilla se genera la misma transformación en ambos modelos.

El ciclo PyTorch es común a ambos modelos: no se utiliza el trainer por defecto de Ultralytics, cuyos recortes y parámetros podrían cambiar la comparación. La normalización ImageNet se incorpora dentro del adaptador VGG; YOLO recibe valores [0,1], conforme a su entrada. Esa diferencia se declara como parte de la adaptación de los pesos, no como aumento de datos. La cabeza YOLO entrega logits para entropía cruzada.

Se inicia un proceso Python nuevo por ejecución para liberar modelos, optimizadores y memoria. No se borran variables de entorno del sistema. La semilla y el estado se inicializan en cada proceso; las advertencias de operaciones no deterministas deben conservarse en los registros.

## Salidas después de ejecutar
- `experiment/<modelo>/seed_<semilla>/`: checkpoint, curvas, métricas, predicciones, errores y matriz de confusión.
- Grad-CAM de VGG para la primera semilla fijada previamente: hasta seis errores y dos aciertos en orden de archivo. No se escoge el modelo por su resultado de prueba.
- `experiment/results.xlsx`, CSV de todas las ejecuciones y resumen media ± DE muestral.
- Prueba exacta de permutación emparejada por semilla para F1 macro. Su inferencia se limita a la variabilidad de entrenamiento con esta partición fija. Con cinco pares, el mínimo p bilateral es 0.0625: no esperar significación a 0.05. Se puede fijar diez semillas **antes** del experimento si hay recursos; no ampliar buscando un p conveniente.
- Latencia FP32 de la red y su adaptador, batch=1, entrada precargada: 20 calentamientos y 100 mediciones sincronizadas. No incluye lectura, transformación ni transferencias. El resumen rechaza mediciones hechas con modelos de GPU diferentes; evaluar todos en una misma sesión.
- El tiempo incluye entrenamiento y validación, excluye descarga inicial de pesos y evaluación de prueba. El tamaño de lote y las épocas iguales no implican igual tiempo de cómputo: se mide esa diferencia.

## Si Colab se interrumpe
Repetir las celdas de instalación y preparación; las ejecuciones completadas se omiten si coinciden configuración y manifiesto. Una ejecución incompleta debe repetirse en una carpeta de resultados nueva o apartando manualmente su carpeta parcial. No hay reanudación a media época. No modificar configuración y mezclar ejecuciones. Guardar todos los checkpoints VGG requiere varios GB en Drive.

## Qué falta para entregar
Ejecutar en GPU, revisar la prueba corta, analizar los errores, completar cifras reales, autores, afiliación y repositorio, y convertir el borrador LaTeX en un artículo final de 6–8 páginas. No se han entrenado los modelos en el computador de preparación: carece de PyTorch y GPU de entrenamiento configurada. La sintaxis y el manifiesto de datos sí se verifican localmente. El notebook realiza la comprobación de ejecución en Colab.

## Limitaciones para el informe (versión breve)
Las particiones agrupan duplicados y variantes detectables; no hay identificadores de paciente verificados y se conservan aumentos preexistentes. Las clases se reportan con sus nombres originales, sin asignarles umbrales de Cobb no documentados.

## Comandos fuera del notebook
Desde esta carpeta, con dependencias y GPU:

```bash
python data/prepare_classification.py --dataset /ruta/scoliosis_depurado --output data
python src/run_all.py --stage smoke --dataset /ruta/scoliosis_depurado --results /ruta/resultados
python src/run_all.py --stage train --dataset /ruta/scoliosis_depurado --results /ruta/resultados
python src/run_all.py --stage evaluate --dataset /ruta/scoliosis_depurado --results /ruta/resultados
python src/summarize.py --results /ruta/resultados
```
