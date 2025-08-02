@echo off
setlocal EnableDelayedExpansion

chcp 65001 >nul

if "%GPU_BACKEND%"=="" set "GPU_BACKEND=NVIDIA"

:MAIN_MENU
cls
echo =====================================
echo    S H A 1    M I N E R    B U I L D
echo =====================================
echo    Current Backend: %GPU_BACKEND%
echo =====================================
echo.
echo [1] Configure Project
echo [2] Build Project
echo [3] Test Project
echo [4] Clean Build Directory
echo [5] Switch GPU Backend
echo [6] Exit
echo.
set /p "choice=Select option [1-6]: "
echo.
if "%choice%"=="1" goto CONFIGURE
if "%choice%"=="2" goto BUILD
if "%choice%"=="3" goto TEST
if "%choice%"=="4" goto CLEAN
if "%choice%"=="5" goto SWITCH_BACKEND
if "%choice%"=="6" goto EXIT
goto MAIN_MENU

:SWITCH_BACKEND
cls
echo =====================================
echo    Select GPU Backend
echo =====================================
echo    Current backend: %GPU_BACKEND%
echo =====================================
echo.
echo [1] NVIDIA (CUDA)
echo [2] AMD (HIP/ROCm)
echo [3] Back to Main Menu
echo.
set /p "backend_choice=Select GPU backend [1-3]: "
echo.
if "%backend_choice%"=="1" set "GPU_BACKEND=NVIDIA"
if "%backend_choice%"=="2" set "GPU_BACKEND=AMD"
if "%backend_choice%"=="3" goto MAIN_MENU
if not "%backend_choice%"=="1" if not "%backend_choice%"=="2" if not "%backend_choice%"=="3" (
    echo Invalid selection!
    pause
)
goto MAIN_MENU

:CONFIGURE
cls
echo =====================================
echo    Configure Presets
echo =====================================
echo.
call :SETUP_VS_ENV
if errorlevel 1 goto MAIN_MENU
if "%GPU_BACKEND%"=="NVIDIA" goto configure_nvidia
if "%GPU_BACKEND%"=="AMD" goto configure_amd
goto MAIN_MENU

:configure_nvidia
    echo NVIDIA/CUDA Build Options:
    echo [1] Windows Ninja Release
    echo [2] Windows Ninja Debug
    echo [3] Windows Visual Studio Release
    echo [4] Windows Visual Studio Debug
    echo [5] Back to Main Menu
    echo.
    set /p "config_choice=Select configuration [1-5]: "
    echo.
    set "PRESET_NAME="
    if "!config_choice!"=="1" set "PRESET_NAME=windows-ninja-release"
    if "!config_choice!"=="2" set "PRESET_NAME=windows-ninja-debug"
    if "!config_choice!"=="3" set "PRESET_NAME=windows-vs2022-release"
    if "!config_choice!"=="4" set "PRESET_NAME=windows-vs2022-debug"
    if "!config_choice!"=="5" goto MAIN_MENU
    if not defined PRESET_NAME echo Invalid selection! & pause & goto CONFIGURE
    echo Configuring project with preset: !PRESET_NAME!
    cmake --preset !PRESET_NAME!
    if errorlevel 1 echo Configuration failed! & pause & goto MAIN_MENU
    echo Configuration successful!
    pause
    goto MAIN_MENU

:configure_amd
    echo AMD/HIP Build Options:
    echo [1] Windows AMD Release (HIP/ROCm)
    echo [2] Windows AMD Debug (HIP/ROCm)
    echo [3] Back to Main Menu
    echo.
    set /p "config_choice=Select configuration [1-3]: "
    echo.
    set "PRESET_NAME="
    if "!config_choice!"=="1" set "PRESET_NAME=windows-hip-release"
    if "!config_choice!"=="2" set "PRESET_NAME=windows-hip-debug"
    if "!config_choice!"=="3" goto MAIN_MENU
    if not defined PRESET_NAME echo Invalid selection! & pause & goto CONFIGURE
    echo Checking AMD/ROCm tools...
    call :CHECK_AMD_TOOLS
    if errorlevel 1 goto MAIN_MENU
    echo Configuring project with preset: !PRESET_NAME!
    cmake --preset !PRESET_NAME!
    if errorlevel 1 echo Configuration failed! & pause & goto MAIN_MENU
    echo Configuration successful!
    pause
    goto MAIN_MENU

:BUILD
cls
echo =====================================
echo    BUILD PRESETS
echo =====================================
echo.
if "%GPU_BACKEND%"=="NVIDIA" goto build_nvidia
if "%GPU_BACKEND%"=="AMD" goto build_amd
goto MAIN_MENU

:build_nvidia
    echo NVIDIA/CUDA Build Options:
    echo [1] Windows Ninja Release
    echo [2] Windows Ninja Debug
    echo [3] Windows Visual Studio Release
    echo [4] Windows Visual Studio Debug
    echo [5] Back to Main Menu
    echo.
    set /p "build_choice=Select build configuration [1-5]: "
    echo.
    set "PRESET_NAME="
    if "!build_choice!"=="1" set "PRESET_NAME=windows-ninja-release"
    if "!build_choice!"=="2" set "PRESET_NAME=windows-ninja-debug"
    if "!build_choice!"=="3" set "PRESET_NAME=windows-vs2022-release"
    if "!build_choice!"=="4" set "PRESET_NAME=windows-vs2022-debug"
    if "!build_choice!"=="5" goto MAIN_MENU
    if not defined PRESET_NAME echo Invalid selection! & pause & goto BUILD
    echo Building with preset: !PRESET_NAME!
    cmake --build --preset !PRESET_NAME!
    if errorlevel 1 echo Build failed! & pause & goto MAIN_MENU
    echo Build successful!
    pause
    goto MAIN_MENU

:build_amd
    echo AMD/HIP Build Options:
    echo [1] Windows AMD Release
    echo [2] Windows AMD Debug
    echo [3] Back to Main Menu
    echo.
    set /p "build_choice=Select build configuration [1-3]: "
    echo.
    set "PRESET_NAME="
    if "!build_choice!"=="1" set "PRESET_NAME=windows-hip-release"
    if "!build_choice!"=="2" set "PRESET_NAME=windows-hip-debug"
    if "!build_choice!"=="3" goto MAIN_MENU
    if not defined PRESET_NAME echo Invalid selection! & pause & goto BUILD
    echo Building with preset: !PRESET_NAME!
    cmake --build --preset !PRESET_NAME!
    if errorlevel 1 echo Build failed! & pause & goto MAIN_MENU
    echo Build successful!
    pause
    goto MAIN_MENU

:TEST
cls
echo =====================================
echo    TEST PRESETS
echo =====================================
echo.
echo [1] Test latest NVIDIA build
echo [2] Test latest AMD build
echo [3] Back to Main Menu
echo.
set /p "test_choice=Select test option [1-3]: "
echo.
if "%test_choice%"=="1" (
    echo Running tests for NVIDIA/CUDA build...
    cmake --test --preset test-windows-release
) else if "%test_choice%"=="2" (
    echo Running tests for AMD/HIP build...
    cmake --test --preset test-windows-hip-release
) else if "%test_choice%"=="3" (
    goto MAIN_MENU
) else (
    echo Invalid selection!
)
pause
goto MAIN_MENU

:CLEAN
cls
echo =====================================
echo    CLEAN BUILD DIRECTORIES
echo =====================================
echo.
echo [1] Clean NVIDIA Ninja Release
echo [2] Clean NVIDIA Ninja Debug
echo [3] Clean NVIDIA VS2022 Release
echo [4] Clean NVIDIA VS2022 Debug
echo [5] Clean AMD/HIP Release
echo [6] Clean AMD/HIP Debug
echo [7] Clean all builds
echo [8] Back to Main Menu
echo.
set /p "clean_choice=Select directory to clean [1-8]: "
echo.
if "%clean_choice%"=="1" rmdir /s /q "build\ninja-release" & echo Cleaning build\ninja-release...
if "%clean_choice%"=="2" rmdir /s /q "build\ninja-debug" & echo Cleaning build\ninja-debug...
if "%clean_choice%"=="3" rmdir /s /q "build\vs2022-release" & echo Cleaning build\vs2022-release...
if "%clean_choice%"=="4" rmdir /s /q "build\vs2022-debug" & echo Cleaning build\vs2022-debug...
if "%clean_choice%"=="5" rmdir /s /q "build\hip-release" & echo Cleaning build\hip-release...
if "%clean_choice%"=="6" rmdir /s /q "build\hip-debug" & echo Cleaning build\hip-debug...
if "%clean_choice%"=="7" rmdir /s /q "build" & echo Cleaning all build directories...
if "%clean_choice%"=="8" goto MAIN_MENU
echo Done!
pause
goto MAIN_MENU

:SETUP_VS_ENV
if defined VSCMD_VER (
    echo Visual Studio environment already configured
    exit /b 0
)
echo Setting up Visual Studio environment...
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" (
            call "%%i\VC\Auxiliary\Build\vcvars64.bat"
            if errorlevel 1 (
                echo ERROR: Failed to initialize Visual Studio environment
                pause
                exit /b 1
            )
            echo Visual Studio environment loaded from: %%i
            exit /b 0
        )
    )
)
echo ERROR: Visual Studio 2022 not found!
echo Install Visual Studio 2022 with C++ tools or run from Developer Command Prompt.
pause
exit /b 1

:CHECK_AMD_TOOLS
echo Checking AMD/ROCm tools...
set "ROCM_FOUND=0"
if defined ROCM_PATH if exist "%ROCM_PATH%\bin\hipcc.exe" set "ROCM_FOUND=1"
if "%ROCM_FOUND%"=="0" (
    if exist "C:\Program Files\AMD\ROCm" (
        for /d %%v in ("C:\Program Files\AMD\ROCm\*") do (
            if exist "%%v\bin\hipcc.exe" (
                set "ROCM_PATH=%%v"
                set "ROCM_FOUND=1"
                echo Found ROCm at: %%v
            )
        )
    )
)
if "%ROCM_FOUND%"=="0" (
    if exist "C:\Program Files\AMD\ROCm\bin\hipcc.exe" (
        set "ROCM_PATH=C:\Program Files\AMD\ROCm"
        set "ROCM_FOUND=1"
        echo Found ROCm at: C:\Program Files\AMD\ROCm
    )
)
if "%ROCM_FOUND%"=="0" (
    for /f "tokens=2*" %%a in ('reg query "HKLM\SOFTWARE\AMD\ROCm" /v InstallDir 2^>nul') do (
        if exist "%%b\bin\hipcc.exe" (
            set "ROCM_PATH=%%b"
            set "ROCM_FOUND=1"
            echo Found ROCm via registry at: %%b
        )
    )
)
if "%ROCM_FOUND%"=="0" (
    echo ERROR: ROCm not found!
    echo Please install ROCm from: https://rocm.docs.amd.com/en/latest/deploy/windows/index.html
    exit /b 1
)
echo AMD/ROCm tools are functional.
exit /b 0

:EXIT
echo Exiting build system...
exit /b 0
