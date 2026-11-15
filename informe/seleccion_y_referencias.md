# Selección de la CNN y referencias verificadas

## Referencia elegida
**Baymurat et al. (2024), “Scoliosis Screening from Plain Radiographs Including Chest and Abdominal X Rays by Using the Deep Learning Method: Is It Worth?”**, J. Turk. Spinal Surg., 35(2), 49–54. [Artículo completo](https://www.jtss.org/articles/doi/jtss.galenos.2024.41736). DOI: 10.4274/jtss.galenos.2024.41736.

El trabajo compara VGG-16, ResNet-101 y GoogLeNet preentrenadas. Usa 1006 radiografías y tres categorías; VGG-16 alcanza 90.1% de exactitud. Reporta SGD con momentum, lote 16, aprendizaje inicial 3×10⁻⁴, dropout 0.5 y L2 0.004. Es una referencia pertinente para elegir una CNN de clasificación con descripción de entrenamiento accesible. No se verificó un repositorio oficial de ese estudio.

## Decisión para este proyecto
Usar **VGG-16 frente a YOLO26n-cls**, ambos clasificadores. Se conserva cada radiografía completa. Las 5299 imágenes depuradas tienen una sola clase distinta por imagen, incluso cuando hay más de una caja; no hace falta inventar una regla para resolver categorías múltiples. No se fusionarán grados ni se les asignarán rangos de Cobb.

La propuesta es una adaptación experimental sobre otro conjunto de datos, con cuatro salidas. No basta para afirmar que se reprodujo la cifra publicada. Si el docente exige reproducir exactamente el conjunto y protocolo originales, hay que ajustar ese alcance antes de presentar la comparación como reproducción literal.

## Parámetros que son decisiones propias
224×224 con preservación de proporción, imágenes completas, 50 épocas máximas, paciencia 10, momentum numérico 0.9 y aumentos ±8° con reflejo 0.5. La partición agrupada 70/25/5 y el ciclo común son decisiones del estudio. Se documentan en lugar de atribuirlas al artículo.

## Trabajos relacionados y pertinencia
| Fuente | Aporte para la revisión | Diferencia respecto a nuestro estudio |
|---|---|---|
| Fraiwan et al., 2022 [2] | Transferencia de aprendizaje sobre radiografías para patologías de columna. | Las clases incluyen escoliosis y espondilolistesis; no equivalen a nuestros cuatro nombres. |
| Tavana et al., 2023 [3] | Clasifica curvaturas con redes preentrenadas y métodos clásicos. | Predice forma C/S, no grados de nuestro dataset. |
| Mulford et al., 2024 [4] | EfficientNet-B6 para categorizar radiografías pediátricas. | Diez categorías radiográficas preoperatorias/posoperatorias. |
| Oh et al., 2024 [5] | Evaluación clínica de medición automatizada de Cobb mediante CNN. | Medición angular, no clasificación de cuatro etiquetas. |
| Zhang et al., 2026 [6] | Compara una estrategia YOLOv11 y otra ResNet sobre radiografías. | Incluye localización/ROI y clasificación binaria; nuestro protocolo utiliza entradas comunes. |

## Referencias en formato IEEE
Las primeras seis son investigaciones del periodo 2022–2026: satisfacen al menos cinco fuentes recientes para una entrega en 2026. Las fuentes técnicas y el dataset complementan la revisión. Comprobar el estado editorial de todas las fuentes de nuevo antes de la entrega definitiva.

[1] A. C. Baymurat et al., “Scoliosis screening from plain radiographs including chest and abdominal X rays by using the deep learning method: Is it worth?,” *J. Turk. Spinal Surg.*, vol. 35, no. 2, pp. 49–54, 2024, doi: [10.4274/jtss.galenos.2024.41736](https://doi.org/10.4274/jtss.galenos.2024.41736).

[2] M. Fraiwan, Z. Audat, L. Fraiwan, and T. Manasreh, “Using deep transfer learning to detect scoliosis and spondylolisthesis from x-ray images,” *PLOS ONE*, vol. 17, no. 5, Art. no. e0267851, 2022, doi: [10.1371/journal.pone.0267851](https://doi.org/10.1371/journal.pone.0267851).

[3] P. Tavana, M. Akraminia, A. Koochari, and A. Bagherifard, “Classification of spinal curvature types using radiography images: Deep learning versus classical methods,” *Artif. Intell. Rev.*, vol. 56, pp. 13259–13291, 2023, doi: [10.1007/s10462-023-10480-w](https://doi.org/10.1007/s10462-023-10480-w).

[4] K. L. Mulford et al., “Deep learning classification of pediatric spinal radiographs for use in large scale imaging registries,” *Spine Deform.*, vol. 12, no. 6, pp. 1607–1614, 2024, doi: [10.1007/s43390-024-00933-9](https://doi.org/10.1007/s43390-024-00933-9).

[5] H. Y. Oh, T. K. Kim, Y. S. Choi, M. Park, R. G. Yoon, and J. K. An, “Radiographic analysis of scoliosis using convolutional neural network in clinical practice,” *J. Korean Soc. Radiol.*, vol. 85, no. 5, pp. 926–936, 2024, doi: [10.3348/jksr.2023.0111](https://doi.org/10.3348/jksr.2023.0111).

[6] H. Zhang, J. Qi, and Z. Zhang, “Spinal x-ray based scoliosis diagnosis using deep learning: A comparison of YOLOv11 and ResNet,” *Biomed. Eng. / Biomed. Tech.*, vol. 71, no. 4–5, pp. 519–535, 2026, doi: [10.1515/bmt-2026-0192](https://doi.org/10.1515/bmt-2026-0192).

[7] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in *ICLR*, 2015. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556).

[8] R. R. Selvaraju et al., “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in *ICCV*, 2017. [Texto original](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf).

[9] Ultralytics, “Image classification with Ultralytics YOLO,” documentación técnica, consultado: 14 nov. 2026. [Documentación](https://docs.ultralytics.com/tasks/classify/).

[10] salmankey, “Balanced Scoliosis X-ray Dataset (YOLOv5 Format),” Kaggle, consultado: 14 nov. 2026. [Dataset](https://www.kaggle.com/datasets/salmankey/balanced-scoliosis-x-ray-dataset-yolov5-format). La copia suministrada declara origen Roboflow y licencia CC BY 4.0 en sus README; no se atribuye a un hospital sin evidencia.

[11] SciPy, “scipy.stats.permutation_test,” documentación técnica, consultado: 14 nov. 2026. [Documentación](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).

[12] TorchVision, “vgg16,” documentación técnica, consultado: 14 nov. 2026. [Documentación](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html).

## Qué buscar dentro de cada fuente
Leer método y conclusiones, identificar la tarea exacta y separar resultados publicados de los nuestros. No citar exactitud de detección o medición angular como si fuera F1 de clasificación. El borrador incorpora solo afirmaciones respaldadas; los números experimentales se añadirán tras ejecutar.
