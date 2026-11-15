# Clasificación de radiografías de escoliosis: VGG-16 frente a YOLO26n-cls

Autor: Yefrey Sneyder Sanchez Alba  
Asignatura: Inteligencia artificial  
Universidad: Universidad Simón Bolívar, sede Cúcuta

## Resultados ejecutados
Se completaron cinco semillas por modelo en Google Colab con GPU Tesla T4. Los resultados de prueba son reales y se encuentran en `results/experiment/`.

| Métrica, media ± DE | VGG-16 | YOLO26n-cls |
|---|---:|---:|
| F1 macro (%) | 68,46 ± 1,07 | 65,50 ± 2,73 |
| Exactitud (%) | 69,62 ± 0,98 | 65,85 ± 2,55 |
| Latencia de red (ms) | 9,15 ± 0,19 | 5,35 ± 0,07 |

La prueba exacta emparejada dio p = 0,0625. VGG obtuvo mayor F1 medio observado; YOLO utilizó menos recursos. No se afirma superioridad estadística a 0,05. La referencia VGG es una adaptación, no una reproducción exacta de sus datos y categorías.

## Estructura
- `data/`: script de clasificación, manifiesto y auditoría; `cleaning/` conserva scripts y registros de depuración y partición.
- `src/`: adaptadores, entrenamiento, evaluación y resumen.
- `results/experiment/`: diez ejecuciones, CSV, matrices, mapas Grad-CAM y Excel.
- `informe/`: artículo en PDF, fuente IEEEtran en LaTeX, texto Markdown y figuras.
- `Colab_escoliosis.ipynb`: notebook de ejecución.
- `config.json`: protocolo realmente utilizado.
- `entorno_registrado.json`: versiones y dispositivo registrados.
- `GUIA_EJECUCION.md`: detalle del ciclo común y del procedimiento en Colab.

## Datos
Fuente: [Balanced Scoliosis X-ray Dataset (YOLOv5 Format)](https://www.kaggle.com/datasets/salmankey/balanced-scoliosis-x-ray-dataset-yolov5-format).
La copia suministrada declara origen Roboflow y licencia CC BY 4.0; se incluyen sus README originales. El repositorio no contiene las imágenes ni los pesos de los modelos. El ZIP depurado se conserva como entrega separada. Cuenta con 5299 imágenes: 3710 de entrenamiento, 265 de validación y 1324 de prueba.

Las particiones se mantienen por grupos detectables de origen/similitud. No hay identificadores de paciente verificados y se conservan aumentos preexistentes. No asignar rangos clínicos de Cobb a las etiquetas originales sin documentación de origen.

## Reproducir

[Abrir en Google Colab](https://colab.research.google.com/github/Sneyderx/comparacion-escoliosis-vgg16-yolo/blob/main/Colab_escoliosis.ipynb). El notebook conecta Drive y clona este repositorio automáticamente; no necesita cargar el ZIP pequeño del proyecto. El ZIP de imágenes depuradas debe estar en Mi unidad de Drive, como indica el notebook.

Usar una GPU, una pareja compatible de PyTorch/TorchVision y las dependencias de `requirements.txt`. El notebook registra `pip freeze` en Drive. Consultar `entorno_registrado.json`; ese archivo no sustituye el listado completo de dependencias.

Desde la raíz del proyecto, suponiendo el dataset depurado extraído en `/ruta/scoliosis_depurado`:

```bash
python data/prepare_classification.py --dataset /ruta/scoliosis_depurado --output data
python src/run_all.py --stage smoke --dataset /ruta/scoliosis_depurado --results /ruta/resultados_nuevos
python src/run_all.py --stage train --dataset /ruta/scoliosis_depurado --results /ruta/resultados_nuevos
python src/run_all.py --stage evaluate --dataset /ruta/scoliosis_depurado --results /ruta/resultados_nuevos
python src/summarize.py --results /ruta/resultados_nuevos
```

La prueba corta no es un resultado experimental. `evaluate` exige que hayan terminado los diez entrenamientos. Para no mezclar archivos nuevos con los entregados, usar una carpeta de resultados nueva.

Para repetir también la limpieza a partir del ZIP original:

```bash
python data/cleaning/prepare_dataset.py --input /ruta/archive.zip --output outputs/scoliosis_depurado
python data/cleaning/refine_similarity.py outputs/scoliosis_depurado
```

Ejecutar el segundo paso de limpieza una sola vez. Los scripts conservan trazabilidad y excluyen anotaciones incompatibles sin inventar una etiqueta correcta.

## Artículo y referencias
El PDF contiene seis páginas de contenido y una de referencias, con doce fuentes. La fuente `.tex` utiliza IEEEtran; puede cargarse en Overleaf junto con la carpeta `figures/`. El PDF entregado fue maquetado a dos columnas y revisado visualmente; la fuente LaTeX no se compiló en el entorno de preparación, por lo que su paginación debe comprobarse al compilarla. El artículo incluye la universidad y el enlace del repositorio.

Referencia de la CNN: A. C. Baymurat et al., J. Turk. Spinal Surg., 2024, DOI [10.4274/jtss.galenos.2024.41736](https://doi.org/10.4274/jtss.galenos.2024.41736). Lista y justificación en `informe/seleccion_y_referencias.md`.

## Enlaces
[Repositorio GitHub](https://github.com/Sneyderx/comparacion-escoliosis-vgg16-yolo)  
[Abrir notebook en Google Colab](https://colab.research.google.com/github/Sneyderx/comparacion-escoliosis-vgg16-yolo/blob/main/Colab_escoliosis.ipynb)

## Uso de IA
Se utilizó asistencia de IA para apoyar código, análisis y redacción. El autor debe revisar el contenido, las referencias y la interpretación antes de la entrega.
