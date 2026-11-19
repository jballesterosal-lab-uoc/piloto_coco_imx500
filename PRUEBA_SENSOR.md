# Evaluar las 45 imágenes de test en el IMX500

El paquete `prueba_test_imx500.zip` contiene las 45 imágenes de la partición de prueba del piloto, con 15 ejemplos por clase. Cada imagen se ha recortado y redimensionado con el mismo `CocoCrops` usado en la evaluación original. Los PNG permiten inspeccionar los recortes. Los ficheros BIN contienen los tensores que se envían directamente al acelerador del IMX500.

`test/test.json` incluye las etiquetas, los hashes de los tensores y las predicciones individuales FP32 y ONNX. El evaluador recoge una predicción por imagen, calcula F1 macro, exactitud y métricas por clase, y compara las decisiones del sensor con las referencias del ordenador.

## Preparar la Raspberry Pi

Descomprime el ZIP y abre un terminal dentro de `test_sensor`.

```bash
sudo apt update
sudo apt install imx500-all python3-picamera2 python3-numpy
python3 -c "import inspect; from picamera2.devices.imx500 import IMX500; print(inspect.signature(IMX500))"
```

La firma debe incluir `tensor_injection`. Si falta, actualiza Picamera2 y el sistema de cámara desde los paquetes oficiales de Raspberry Pi OS. La implementación usa las mismas APIs de inyección que el [ejemplo oficial de Picamera2](https://github.com/raspberrypi/picamera2-examples/blob/main/examples/imx500/imx500_object_detection_injection_demo.py), incluidas dos APIs privadas cuyo nombre puede cambiar entre versiones.

Sony indica que la inyección requiere desactivar la caché de la AI Camera. Revisa `/boot/firmware/config.txt` y la conexión física de la cámara. Si aún usas la detección automática, sustituye su ajuste por `camera_auto_detect=0` y declara explícitamente la cámara con `bypass-cache`. Para una Pi 5, elige la línea que corresponde al conector utilizado.

| Conector de la cámara | Línea del overlay |
| --- | --- |
| CAM/DISP0 | `dtoverlay=imx500,cam0,bypass-cache` |
| CAM/DISP1 | `dtoverlay=imx500,bypass-cache` |

Si ya existe una línea `dtoverlay=imx500`, añade `bypass-cache` a esa línea conservando el conector. Reinicia la Pi después de guardar el cambio. Puedes editar el fichero con `sudo nano /boot/firmware/config.txt` y reiniciar con `sudo reboot`. [Instrucciones de inyección de Sony](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library/blob/main/docs/src/getting_started/data_injection.md). [Conectores de cámara de Raspberry Pi](https://www.raspberrypi.com/documentation/accessories/camera.html).

## Ejecutar la evaluación

Empaqueta la salida del conversor incluida en el ZIP para evaluar el modelo correspondiente a las referencias.

```bash
mkdir -p modelo_test
imx500-package -i packerOut.zip -o modelo_test -f input_format.json
python3 evaluar_test_sensor.py --model modelo_test/network.rpk \
  --test test --output resultado_sensor_01.json
```

La cámara debe estar conectada. El evaluador introduce los tensores por software y registra cada resultado cuando su contador de fotograma coincide con el de la inyección. Si se pierde ese fotograma, vuelve a enviar la misma muestra hasta un máximo de diez intentos (`--retries`). Si el driver rechaza el tensor con un error transitorio (`Errno 11` o `Errno 121`), repite el envío hasta 50 veces (`--io-retries`). La documentación de inyección de Sony describe estos errores y señala que aumentan con la tasa de fotogramas y el número de inyecciones. La tasa predeterminada es de 12 fotogramas por segundo (`--frame-rate`, mínimo 5). El JSON registra los reintentos de cada muestra. El tiempo de espera predeterminado es de 15 segundos por intento. El arranque permite hasta 180 segundos para cargar el firmware.

Verás una línea por imagen con la etiqueta real y la predicción. Al terminar se generan dos archivos.

- `resultado_sensor_01.json` conserva los logits, las métricas, la matriz de confusión, los tiempos DNN y la concordancia con cada referencia.
- `resultado_sensor_01.csv` permite comparar por imagen la etiqueta real, FP32, ONNX y sensor.

Cada ejecución requiere un nombre de salida nuevo para conservar las anteriores. El JSON se actualiza después de cada imagen y solo lleva `status: complete` cuando se han evaluado las 45. Si falla, guarda `status: failed`, el error y las muestras recibidas hasta ese momento.

## Comparabilidad de las entradas

Las referencias `fp32` y `onnx` usan el recorte RGB normalizado a 0–1 de la evaluación original. El modelo compilado recibe una entrada `uint8` con escala `1/256` y desplazamiento 0, según su `dnnParams.xml`. El generador prepara los bytes mediante `clip(round(x / escala), 0, 255)` y orden planar RGB. El tamaño de 160 × 160 cumple la alineación del IMX500 de 32 píxeles en anchura y 2 en altura.

La referencia adicional `onnx_injected_input` evalúa ONNX sobre los valores reconstruidos de los mismos bytes inyectados. Esto permite medir la concordancia con el sensor sobre la misma rejilla de entrada. El informe registra también la comparación con las referencias originales. Esta prueba caracteriza el modelo compilado y la cadena de inyección. Las capturas ópticas anteriores siguen aportando información sobre la iluminación, el encuadre y el preprocesado de la cámara.

## Regenerar el paquete en el ordenador

Con el entorno del piloto activo y las imágenes originales disponibles, ejecuta desde la raíz del repositorio.

```bash
python preparar_prueba_sensor.py --output resultados/test_sensor
```

Se obtienen `resultados/test_sensor/test` y `resultados/test_sensor/prueba_test_imx500.zip`. El generador comprueba que las matrices de confusión FP32 y ONNX coinciden con las guardadas en el experimento original antes de entregar el ZIP. El directorio de salida debe ser nuevo.
