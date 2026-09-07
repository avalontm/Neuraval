# Neuraval.Cuda.Native

Librería nativa en CUDA C++ que expone una API en C para ser consumida desde C# mediante P/Invoke (proyecto `Neuraval.Cuda`).

## Requisitos

- CUDA Toolkit 11.x o superior (`nvcc` en el `PATH`).
- GPU NVIDIA con arquitectura Pascal (sm_61) o superior.
- Driver NVIDIA compatible con la versión del toolkit instalada.

## Compilación

Antes de compilar, `build.sh`/`build.bat` verifican automáticamente:
- Que haya una GPU NVIDIA con driver instalado (`nvidia-smi`).
- Que `nvcc` esté en el `PATH` (CUDA Toolkit instalado).
- El compilador de C++ que `nvcc` usa por atrás (`cl.exe` en Windows, `gcc`/`g++` en Linux).

Si falta algo, el script corta con un mensaje indicando qué instalar, en vez de seguir e imprimir un mensaje de éxito falso.

### Linux

```bash
./build.sh
```

Genera `output/libnavcuda.so`.

### Windows

```bat
build.bat
```

Genera `output\navcuda.dll`.

### Alternativa con CMake

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Instalación del binario compilado

Copiar el binario generado (`libnavcuda.so` o `navcuda.dll`) al directorio de salida del ejecutable .NET que lo consume (por ejemplo, junto a `Neuraval.CLI.dll` en `bin/Debug/net10.0/` o `bin/Release/net10.0/`), o a cualquier ruta incluida en `LD_LIBRARY_PATH` / `PATH`.

## Estado de esta fase

Esta primera fase cubre:

- Detección de dispositivo CUDA (`ncb_cuda_device_count`, `ncb_cuda_get_device_name`).
- Inicialización de dispositivo (`ncb_cuda_init`).
- Gestión de memoria en el dispositivo (`ncb_cuda_alloc`, `ncb_cuda_free`, copias host↔device).
- Multiplicación de matrices en GPU con kernel tileado (`ncb_cuda_matmul`, `ncb_cuda_matmul_device`).

No cubre todavía: atención multi-cabeza, feed-forward, softmax ni el optimizador Adam en GPU — eso corresponde a las siguientes subfases, una vez validada esta base.

## Nota importante

Este entorno de desarrollo no cuenta con `nvcc` ni una GPU NVIDIA disponible, por lo que el kernel no pudo compilarse ni probarse aquí. El código fue escrito siguiendo la API estándar de CUDA Runtime; se recomienda compilarlo y probarlo en una máquina con el Toolkit instalado antes de integrarlo a un flujo de entrenamiento real.
