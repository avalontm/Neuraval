# Clasificación de imágenes con Neuraval

El CLI puede entrenar un clasificador local de imágenes con una red neuronal feed-forward, guardar sus pesos en un archivo binario `.nvimg`, cargarlo después y predecir o evaluar imágenes etiquetadas. No usa servicios externos ni paquetes de visión adicionales.

## Dataset

Crea una carpeta por clase. El nombre de la carpeta es la etiqueta; las imágenes se buscan también en subcarpetas. Se aceptan PNG de 8 bits no entrelazados en escala de grises/RGB/RGBA y BMP sin compresión de 24 o 32 bits.

```text
dataset/
  gatos/
    gato-01.png
    gato-02.bmp
  perros/
    perro-01.png
    perro-02.bmp
```

Para números, usa carpetas `0` a `9`. Se recomienda tener varias imágenes variadas por clase. El demo convierte a escala de grises y redimensiona al tamaño elegido, que por defecto es 28 × 28.

## Entrenar, probar y predecir

```powershell
dotnet run --project Neuraval.CLI -- vision train --data .\dataset\train --output .\modelos\mascotas.nvimg --size 28 --epochs 10 --hidden 128 --cpu
dotnet run --project Neuraval.CLI -- vision test --model .\modelos\mascotas.nvimg --data .\dataset\test
dotnet run --project Neuraval.CLI -- vision predict --model .\modelos\mascotas.nvimg --image .\foto.png
```

`--cpu` evita usar CUDA en este demo. Entrenamiento imprime pérdida y precisión sobre los mismos datos usados para entrenar; `vision test` calcula la precisión en otra carpeta etiquetada. Ajusta `--size`, `--hidden`, `--epochs` y `--learning-rate` según el tamaño del dataset y el equipo.

Este primer clasificador es una red densa, no una CNN. Sirve para probar el ciclo de datos, entrenamiento y exportación; imágenes complejas como gatos y perros pueden requerir más ejemplos y una arquitectura convolucional para alcanzar buena precisión.
