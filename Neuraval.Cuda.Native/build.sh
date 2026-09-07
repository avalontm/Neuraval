#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"

echo "Verificando requisitos..."
echo

# --- 1. GPU NVIDIA presente y driver instalado ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[FALTA] nvidia-smi no encontrado: no se detecta un driver de GPU NVIDIA instalado."
    echo "        Si tenés una GPU NVIDIA, instalá el driver de tu distro o desde nvidia.com/drivers."
    echo "        Si no tenés GPU NVIDIA, este proyecto no puede compilarse en esta máquina."
    exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
    echo "[FALTA] nvidia-smi está pero falló al ejecutarse: revisá el driver de tu GPU."
    exit 1
fi
echo "[OK] GPU NVIDIA y driver detectados."

# --- 2. nvcc (CUDA Toolkit) en el PATH ---
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[FALTA] no se encontró 'nvcc' en el PATH."
    echo "        Instalá el CUDA Toolkit (https://developer.nvidia.com/cuda-downloads) y"
    echo "        asegurate de que su carpeta bin (normalmente /usr/local/cuda/bin) esté en el PATH."
    exit 1
fi
echo "[OK] nvcc encontrado."

# --- 3. Compilador de C++ (gcc/g++): nvcc lo necesita por atrás en Linux ---
if ! command -v gcc >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
    echo "[FALTA] no se encontró gcc/g++. nvcc los necesita para compilar en Linux."
    echo "        Instalalos con: sudo apt-get install build-essential"
    exit 1
fi
echo "[OK] compilador de C++ (gcc/g++) encontrado."
echo

mkdir -p "$OUTPUT_DIR"

echo "Compilando navcuda.cu..."
nvcc -O3 -shared -Xcompiler -fPIC \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_86,code=sm_86 \
    -o "$OUTPUT_DIR/libnavcuda.so" \
    "$SCRIPT_DIR/src/navcuda.cu"

if [ ! -f "$OUTPUT_DIR/libnavcuda.so" ]; then
    echo "ERROR: nvcc no reportó error pero libnavcuda.so no se generó. Revisá el log de arriba."
    exit 1
fi

echo
echo "Librería generada en $OUTPUT_DIR/libnavcuda.so"
echo "Copiala junto a Neuraval.CLI.dll (o a una ruta en LD_LIBRARY_PATH) antes de"
echo "correr 'dotnet run --project Neuraval.CLI'."
