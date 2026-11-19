# Piloto de clasificación COCO en Sony IMX500

Este experimento previo del TFM reproduce la cadena técnica que luego se aplicará a la clasificación de expresiones faciales con ABAW 8th. Sigue el ejemplo [`mobilenet_v2_card.ini` de Sony](https://github.com/SonySemiconductorSolutions/aitrios-rpi-training-samples/blob/main/samples/mobilenet_v2_card.ini) en la elección de MobileNetV2, PyTorch, aprendizaje por transferencia, entrada de 160 × 160 y cuantización con MCT. El código ejecuta el flujo con [Edge-MDT 1.3.0](https://pypi.org/project/edge-mdt/1.3.0/) en un entorno propio.

## Resultados del piloto

| Modelo | F1 macro | Exactitud | Aciertos |
| --- | ---: | ---: | ---: |
| FP32 | 0,8912 | 88,89 % | 40/45 |
| PTQ en ONNX Runtime | 0,9117 | 91,11 % | 41/45 |

El checkpoint seleccionado corresponde a la época 5. La diferencia de F1 macro fue de +0,0205 en esta ejecución. Con 45 ejemplos de prueba, se interpreta como un resultado descriptivo del piloto. El conversor IMX500 3.18.2 informó un uso de memoria de 3,11 MB sobre 8 MB (39 %) y confirmó que el modelo cabe en el sensor con el tensor de entrada persistente. [Métricas FP32](resultados/piloto/fp32.json) · [Métricas PTQ](resultados/piloto/ptq.json) · [Informe de memoria](resultados/piloto/conversion/imx500_MemoryReport.json) · [Versiones del entorno](resultados/piloto/entorno.txt).

Para ir directamente a la cámara, descarga el [ZIP de despliegue](resultados/piloto/despliegue_pi.zip) y sigue los pasos 3 y 4 en la Raspberry Pi. El ZIP contiene la salida del conversor que se empaqueta como `network.rpk` en la Pi.

Para enviar al sensor las 45 imágenes del conjunto de prueba y comparar sus resultados con FP32 y ONNX, sigue la [guía de evaluación mediante inyección](PRUEBA_SENSOR.md). El [paquete de test](resultados/test_sensor/prueba_test_imx500.zip) incluye los recortes, los tensores, las etiquetas y las referencias del ordenador.

## Datos y protocolo

La tarea asigna una etiqueta (`bottle`, `banana` o `book`, en ese orden) a un recorte RGB. El recorte se obtiene de las cajas de [COCO 2017](https://cocodataset.org/#download). Se descartan las cajas `iscrowd` y las que, tras recortarlas a los bordes, miden menos de 24 píxeles en algún lado o no caben en un cuadrado dentro de la imagen. Por imagen se conserva la caja válida de mayor área entre las clases seleccionadas. El recorte cuadrado añade aproximadamente un 10 % al lado dominante cuando el tamaño de la foto lo permite. El modelo recibe el recorte redimensionado a 160 × 160 y valores RGB de 0 a 1. En cámara, el objeto se coloca dentro de una región cuadrada central y el sensor clasifica ese encuadre.

| Partición del piloto | Por clase | Total |
| --- | ---: | ---: |
| Entrenamiento | 45 | 135 |
| Validación | 10 | 30 |
| Calibración para PTQ | 10 | 30 |
| Prueba | 15 | 45 |

Las cuatro particiones son estratificadas y disjuntas por `image_id`. Todas proceden de `val2017`, para reutilizar las anotaciones disponibles y reducir el coste. Con solo 45 imágenes de prueba, la evaluación sirve para comparar FP32 y PTQ en las mismas entradas y documentar el piloto. Las métricas dependen de los recortes proporcionados por las anotaciones y del subconjunto de clases.

## 1. Preparar el entorno y las imágenes

Ejecuta desde la carpeta `piloto_coco_imx500` en Linux. Sony documenta el conversor en Linux con Java 17 y Python 3.11 como referencia. En Windows se puede ejecutar desde Ubuntu en WSL con Python 3.11 o 3.12. Los comandos siguientes usan las ruedas de PyTorch para CPU. Para entrenar con GPU, usa las ruedas PyTorch 2.5.1 y TorchVision 0.20.1 adecuadas a la GPU. [Manual de Edge-MDT](https://github.com/SonySemiconductorSolutions/aitrios-edge-mdt). [Versiones de PyTorch](https://pytorch.org/get-started/previous-versions/).

```bash
python3 -m venv ~/.venvs/tfm-coco-imx500
source ~/.venvs/tfm-coco-imx500/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements.txt
java -version
python -m pip check

curl -fL https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip \
  -o datos/annotations_trainval2017.zip
python preparar_coco.py --annotations datos/annotations_trainval2017.zip \
  --output datos --download-to datos/val2017
```

En WSL, el entorno virtual se instala en el sistema de archivos de Linux para acelerar la instalación. Si ya tienes las imágenes originales, omite `--download-to` y proporciona su carpeta al entrenar y cuantizar mediante `--images /ruta/al/val2017`. `--annotations` acepta también `instances_val2017.json` o un ZIP que lo contenga, como `coco2017labels.zip`. [Manifiesto de las muestras](datos/muestras.csv) · [Semilla y SHA-256 de las anotaciones](datos/dataset.json).

## 2. Entrenar y cuantizar

```bash
mkdir -p resultados/reproduccion
python -m pip freeze > resultados/reproduccion/entorno.txt
python modelo.py train --device cpu --run resultados/reproduccion
python modelo.py quantize --run resultados/reproduccion
```

MobileNetV2 parte de los pesos `IMAGENET1K_V1`. Durante las dos primeras épocas se entrena la cabeza, y a partir de la tercera se ajustan también los últimos cuatro bloques de `features`, conservando las estadísticas de BatchNorm. Solo el entrenamiento usa volteo horizontal aleatorio. Se elige el checkpoint por F1 macro en validación. La salida son tres logits.

PTQ usa las 30 imágenes de calibración y el perfil IMX500 TPC 4.0. La evaluación carga el ONNX exportado en ONNX Runtime con las extensiones de MCT. `fp32.json` y `ptq.json` registran F1 macro, exactitud, matriz de confusión, precisión, exhaustividad y F1 por clase en las mismas 45 imágenes de prueba. El cambio se registra como `F1 PTQ − F1 FP32`. El ONNX contiene la representación de cuantización que espera el conversor, por lo que su tamaño en disco y el coste en el sensor se interpretan por separado. [Compatibilidad de TPC](https://github.com/SonySemiconductorSolutions/aitrios-edge-mdt-tpc). [Cuantización simulada y exportación](https://github.com/SonySemiconductorSolutions/mct-model-optimization/blob/main/FAQ.md).

## 3. Convertir y empaquetar

En Linux, con el entorno virtual activo:

```bash
imxconv-pt -i resultados/reproduccion/imx500.onnx -o resultados/reproduccion/conversion
```

Guarda `conversion/packerOut.zip` y el informe de memoria. Si el conversor indica que el modelo excede la memoria, prueba `--no-input-persistency` en otro directorio de salida y registra que esa opción desactiva el tensor de entrada para depuración. El ZIP `resultados/piloto/despliegue_pi.zip` ya reúne `packerOut.zip`, `labels.txt`, `input_format.json` y `probar_camara.py`. Cópialo y descomprímelo en la Raspberry Pi. [Conversión y empaquetado de Raspberry Pi](https://www.raspberrypi.com/documentation/accessories/ai-camera.html#model-deployment).

En Raspberry Pi OS, con la AI Camera conectada:

```bash
sudo apt update
sudo apt install imx500-all python3-picamera2
rpicam-hello --list-cameras
mkdir -p camara
imx500-package -i packerOut.zip -o camara -f input_format.json
```

## 4. Registrar la prueba de cámara

Coloca un objeto en el centro de la imagen. Usa el mismo encuadre y repite con las tres clases. Guarda un registro por escena y anota el modelo de Pi, la versión del sistema, la distancia y la iluminación. Añade `--preview` si ejecutas la prueba desde un escritorio de la Raspberry Pi.

```bash
python3 probar_camara.py --model camara/network.rpk --labels labels.txt \
  --expected bottle --output camara/bottle.json
python3 probar_camara.py --model camara/network.rpk --labels labels.txt \
  --expected banana --output camara/banana.json
python3 probar_camara.py --model camara/network.rpk --labels labels.txt \
  --expected book --output camara/book.json
```

El script comprueba que el IMX500 entrega un tensor de tres logits finitos y recoge 60 salidas tras el calentamiento. Guarda las predicciones, la fracción de fotogramas con la clase esperada y las salidas recibidas por segundo. Cuando la cámara publica `CnnKpiInfo`, también registra la mediana y el percentil 95 del tiempo DNN en el sensor. La tasa de salidas mide el ritmo observado de recepción. La fracción de aciertos en fotogramas consecutivos describe la estabilidad de esa escena. Para medir concordancia entre el sensor y el ordenador sobre las mismas imágenes, Sony documenta una evaluación posterior por inyección de datos. [Metadatos del IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html#ai-camera-applications). [Evaluación por inyección de datos](https://github.com/SonySemiconductorSolutions/aitrios-rpi-sandbox/tree/main/examples/evaluation).
