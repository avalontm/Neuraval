@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set OUTPUT_DIR=%SCRIPT_DIR%output

echo Verificando requisitos...
echo.

rem --- 1. GPU NVIDIA presente y driver instalado ---
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [FALTA] nvidia-smi no encontrado: no se detecta un driver de GPU NVIDIA instalado.
    echo         Si tenes una GPU NVIDIA, instala el driver desde nvidia.com/drivers.
    echo         Si no tenes GPU NVIDIA, este proyecto no puede compilarse en esta maquina.
    exit /b 1
)
nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [FALTA] nvidia-smi esta pero fallo al ejecutarse: revisa el driver de tu GPU.
    exit /b 1
)
echo [OK] GPU NVIDIA y driver detectados.

rem --- 2. nvcc (CUDA Toolkit) en el PATH ---
where nvcc >nul 2>nul
if errorlevel 1 (
    echo [FALTA] "nvcc" no encontrado en el PATH.
    echo         Instala el CUDA Toolkit ^(https://developer.nvidia.com/cuda-downloads^)
    echo         y abri una consola NUEVA despues de instalar ^(el PATH no se actualiza
    echo         en consolas ya abiertas^).
    exit /b 1
)
echo [OK] nvcc encontrado.

rem --- 3. cl.exe (Visual C++ Build Tools) en el PATH: nvcc lo necesita en Windows ---
where cl >nul 2>nul
if errorlevel 1 (
    echo [INFO] cl.exe no esta en el PATH todavia. Buscando una instalacion de
    echo        Visual Studio para cargar su entorno x64 automaticamente...

    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if not exist "!VSWHERE!" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"

    if not exist "!VSWHERE!" (
        echo [FALTA] No se encontro vswhere.exe, asi que no se pudo detectar
        echo         Visual Studio automaticamente. Instala Build Tools for
        echo         Visual Studio ^(visualstudio.microsoft.com/downloads^) con el
        echo         workload Desarrollo para el escritorio con C++.
        exit /b 1
    )

    set "VSINSTALL="
    for /f "usebackq tokens=*" %%I in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VSINSTALL=%%I"
    )

    if not defined VSINSTALL (
        echo [FALTA] No se encontro ninguna instalacion de Visual Studio con el
        echo         workload de C++ ^(Microsoft.VisualStudio.Component.VC.Tools.x86.x64^).
        echo         Instala Build Tools for Visual Studio con el workload
        echo         Desarrollo para el escritorio con C++.
        exit /b 1
    )

    set "VCVARS=!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat"
    if not exist "!VCVARS!" (
        echo [FALTA] Se encontro Visual Studio en !VSINSTALL! pero no vcvars64.bat.
        exit /b 1
    )

    echo [OK] Cargando entorno de compilacion desde:
    echo      !VCVARS!
    call "!VCVARS!"

    where cl >nul 2>nul
    if errorlevel 1 (
        echo [FALTA] Se cargo vcvars64.bat pero cl.exe sigue sin aparecer en el PATH.
        echo         Revisa tu instalacion de Visual Studio Build Tools.
        exit /b 1
    )
)
echo [OK] cl.exe encontrado.
echo.

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

rem --- Detectar la capacidad de computo de la GPU para compilar el arch correcto ---
rem     (evita hardcodear archs viejos que versiones nuevas de nvcc ya no soportan)
rem     Usamos "--format=csv" sin "noheader" porque la coma dentro de "csv,noheader"
rem     puede romperse al pasar por el for /f con backticks; en cambio saltamos el
rem     encabezado con "skip=1" del propio for /f.
set "COMPUTE_CAP="
for /f "usebackq skip=1 tokens=* delims=" %%C in (`nvidia-smi --query-gpu=compute_cap --format=csv`) do (
    if not defined COMPUTE_CAP set "COMPUTE_CAP=%%C"
)

if not defined COMPUTE_CAP (
    echo [FALTA] No se pudo detectar la capacidad de computo de la GPU via nvidia-smi.
    exit /b 1
)

set "ARCH=%COMPUTE_CAP:.=%"
echo [OK] GPU con capacidad de computo %COMPUTE_CAP% detectada ^(sm_%ARCH%^).

echo Compilando navcuda.cu...
nvcc -O3 -shared ^
    -gencode arch=compute_%ARCH%,code=sm_%ARCH% ^
    -gencode arch=compute_%ARCH%,code=compute_%ARCH% ^
    -o "%OUTPUT_DIR%\navcuda.dll" ^
    "%SCRIPT_DIR%src\navcuda.cu"

if errorlevel 1 (
    echo ERROR: nvcc fallo al compilar navcuda.cu. Revisa el mensaje de arriba.
    echo        Si el error menciona "Unsupported gpu architecture", tu version de
    echo        nvcc puede ser demasiado vieja para tu GPU ^(sm_%ARCH%^). Actualiza el
    echo        CUDA Toolkit desde developer.nvidia.com/cuda-downloads.
    exit /b 1
)

if not exist "%OUTPUT_DIR%\navcuda.dll" (
    echo ERROR: nvcc no reporto error pero navcuda.dll no se genero. Revisa el log de arriba.
    exit /b 1
)

echo.
echo Libreria generada en %OUTPUT_DIR%\navcuda.dll
echo Copiala junto a Neuraval.CLI.dll ^(por ejemplo Neuraval.CLI\bin\Debug\net10.0\^)
echo antes de correr "dotnet run --project Neuraval.CLI".
endlocal