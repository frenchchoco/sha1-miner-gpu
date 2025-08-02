@echo off
setlocal EnableDelayedExpansion

chcp 65001 >nul

if "%GPU_BACKEND%"=="" set "GPU_BACKEND=NVIDIA"

:MAIN_MENU
cls
echo =====================================
echo    S H A 1   M I N E R   B U I L D
echo =====================================
echo    Current Backend: %GPU_BACKEND%
echo =====================================
echo.
echo [1] Configure Project
echo [2] Build Project
echo [3] Test Project
echo [4] Clean Build Directory
echo [5] Setup vcpkg (and install packages)
echo [6] Switch GPU Backend
echo [7] Exit
echo.
set /p "choice=Select option [1-7]: "
echo.
if "%choice%"=="1" goto CONFIGURE
if "%choice%"=="2" goto BUILD
if "%choice%"=="3" goto TEST
if "%choice%"=="4" goto CLEAN
if "%choice%"=="5" goto SETUP_VCPKG
if "%choice%"=="6" goto SWITCH_BACKEND
if "%choice%"=="7" goto EXIT
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
if "%backend_choice%" neq "1" if "%backend_choice%" neq "2" if "%backend_choice%" neq "3" echo Invalid selection!
if "%backend_choice%" neq "1" if "%backend_choice%" neq "2" if "%backend_choice%" neq "3" pause
if "%backend_choice%" neq "1" if "%backend_choice%" neq "2" if "%backend_choice%" neq "3" goto MAIN_MENU
pause
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
    set "GENERATOR_NAME="
    set "CMAKE_BUILD_TYPE="
    set "BUILD_DIR="
    set "CUDA_ARCH="
    if "!config_choice!"=="1" set "GENERATOR_NAME=Ninja" & set "CMAKE_BUILD_TYPE=Release" & set "BUILD_DIR=build/ninja-release" & set "CUDA_ARCH=50;75;80;86"
    if "!config_choice!"=="2" set "GENERATOR_NAME=Ninja" & set "CMAKE_BUILD_TYPE=Debug" & set "BUILD_DIR=build/ninja-debug" & set "CUDA_ARCH=50"
    if "!config_choice!"=="3" set "GENERATOR_NAME=Visual Studio 17 2022" & set "CMAKE_BUILD_TYPE=Release" & set "BUILD_DIR=build/vs2022-release" & set "CUDA_ARCH=50;75;80;86"
    if "!config_choice!"=="4" set "GENERATOR_NAME=Visual Studio 17 2022" & set "CMAKE_BUILD_TYPE=Debug" & set "BUILD_DIR=build/vs2022-debug" & set "CUDA_ARCH=50"
    if "!config_choice!"=="5" goto MAIN_MENU
    if not defined GENERATOR_NAME echo Invalid selection! & pause & goto CONFIGURE
    echo Checking NVIDIA/CUDA tools...
    set "CUDA_PATH="
    set "CUDA_VERSION=v12.9"
    if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\%CUDA_VERSION%" set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\%CUDA_VERSION%"
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%CUDA_VERSION%" set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%CUDA_VERSION%"
    if not defined CUDA_PATH echo ERROR: Could not find CUDA %CUDA_VERSION% installation. & pause & goto MAIN_MENU
    if not exist "%CUDA_PATH%\bin\nvcc.exe" echo ERROR: nvcc.exe not found in "%CUDA_PATH%\bin". & pause & goto MAIN_MENU
    echo Found CUDA at: %CUDA_PATH%
    set "TEMP_FILE=%TEMP%\nvcc_version_output.txt"
    "%CUDA_PATH%\bin\nvcc.exe" --version > "%TEMP_FILE%" 2>&1
    for /f "tokens=*" %%i in ('findstr /i "release" "%TEMP_FILE%"') do set "NVCC_VERSION_STRING=%%i"
    del "%TEMP_FILE%" >nul 2>&1
    chcp 65001 >nul
    echo CUDA version: %NVCC_VERSION_STRING%
    path "%CUDA_PATH%\bin";%PATH%
    echo NVIDIA/CUDA tools are functional.
    where ninja >nul 2>&1
    if /i "!GENERATOR_NAME!"=="Ninja" if errorlevel 1 echo ERROR: Ninja not found! Ninja is required for this preset. & if errorlevel 1 pause & if errorlevel 1 goto MAIN_MENU
    goto finish_configure_selection

:configure_amd
    echo AMD/HIP Build Options:
    echo [1] Windows AMD Release (HIP/ROCm)
    echo [2] Windows AMD Debug (HIP/ROCm)
    echo [3] Back to Main Menu
    echo.
    set /p "config_choice=Select configuration [1-3]: "
    echo.
    set "GENERATOR_NAME="
    set "CMAKE_BUILD_TYPE="
    set "BUILD_DIR="
    set "HIP_ARCH="
    if "!config_choice!"=="1" set "GENERATOR_NAME=Ninja" & set "CMAKE_BUILD_TYPE=Release" & set "BUILD_DIR=build/hip-release" & set "HIP_ARCH=gfx1030;gfx1100"
    if "!config_choice!"=="2" set "GENERATOR_NAME=Ninja" & set "CMAKE_BUILD_TYPE=Debug" & set "BUILD_DIR=build/hip-debug" & set "HIP_ARCH=gfx1030;gfx1100"
    if "!config_choice!"=="3" goto MAIN_MENU
    if not defined GENERATOR_NAME echo Invalid selection! & pause & goto CONFIGURE
    call :CHECK_AMD_TOOLS
    if errorlevel 1 echo Configuration aborted due to missing AMD/ROCm tools. & pause & goto MAIN_MENU
    goto finish_configure_selection

:finish_configure_selection
if not exist "vcpkg\scripts\buildsystems\vcpkg.cmake" echo ERROR: vcpkg toolchain file not found! Please run option 5. & pause & goto MAIN_MENU
if not exist "!BUILD_DIR!" mkdir "!BUILD_DIR!"
echo Configuring with direct command:
cmake -S . -B "!BUILD_DIR!" -G "!GENERATOR_NAME!" ^
    -D CMAKE_BUILD_TYPE="!CMAKE_BUILD_TYPE!" ^
    -D CMAKE_TOOLCHAIN_FILE="%cd%/vcpkg/scripts/buildsystems/vcpkg.cmake" ^
    -D VCPKG_TARGET_TRIPLET="x64-windows-static" ^
    -D USE_HIP=OFF ^
    -D CMAKE_CUDA_ARCHITECTURES="!CUDA_ARCH!"
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
call :SETUP_VS_ENV
if errorlevel 1 goto MAIN_MENU
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
    set "BUILD_DIR_SEL="
    set "BUILD_CONFIG_SEL="
    if "!build_choice!"=="1" set "BUILD_DIR_SEL=build/ninja-release" & set "BUILD_CONFIG_SEL=Release"
    if "!build_choice!"=="2" set "BUILD_DIR_SEL=build/ninja-debug" & set "BUILD_CONFIG_SEL=Debug"
    if "!build_choice!"=="3" set "BUILD_DIR_SEL=build/vs2022-release" & set "BUILD_CONFIG_SEL=Release"
    if "!build_choice!"=="4" set "BUILD_DIR_SEL=build/vs2022-debug" & set "BUILD_CONFIG_SEL=Debug"
    if "!build_choice!"=="5" goto MAIN_MENU
    if not defined BUILD_DIR_SEL echo Invalid selection! & pause & goto BUILD
    goto finish_build_selection

:build_amd
    echo AMD/HIP Build Options:
    echo [1] Windows AMD Release
    echo [2] Windows AMD Debug
    echo [3] Back to Main Menu
    echo.
    set /p "build_choice=Select build configuration [1-3]: "
    echo.
    set "BUILD_DIR_SEL="
    set "BUILD_CONFIG_SEL="
    if "!build_choice!"=="1" set "BUILD_DIR_SEL=build/hip-release" & set "BUILD_CONFIG_SEL=Release"
    if "!build_choice!"=="2" set "BUILD_DIR_SEL=build/hip-debug" & set "BUILD_CONFIG_SEL=Debug"
    if "!build_choice!"=="3" goto MAIN_MENU
    if not defined BUILD_DIR_SEL echo Invalid selection! & pause & goto BUILD
    goto finish_build_selection

:finish_build_selection
echo Building %GPU_BACKEND% configuration...
echo Build directory: %BUILD_DIR_SEL%
if not exist "%BUILD_DIR_SEL%" echo ERROR: Build directory %BUILD_DIR_SEL% does not exist! & echo Please configure the project first (Option 1). & pause & goto MAIN_MENU
if not exist "%BUILD_DIR_SEL%\CMakeCache.txt" echo ERROR: Project not configured in %BUILD_DIR_SEL% & echo Please configure the project first. & pause & goto MAIN_MENU
cmake --build "%BUILD_DIR_SEL%" --config %BUILD_CONFIG_SEL% --verbose
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
echo [1] Test current configuration
echo [2] Run custom test command
echo [3] Back to Main Menu
echo.
set /p "test_choice=Select test option [1-3]: "
echo.
if "%test_choice%"=="1" (
    echo.
    if "%GPU_BACKEND%"=="AMD" (
        echo Running tests for AMD/HIP build...
        if exist "build\hip-release\sha1_miner.exe" (
            cd build\hip-release
            sha1_miner.exe --test
            cd ..\..
        ) else if exist "build\hip-debug\sha1_miner.exe" (
            cd build\hip-debug
            sha1_miner.exe --test
            cd ..\..
        ) else (
            echo No AMD build found! Build first.
        )
    ) else (
        echo Running tests for NVIDIA/CUDA build...
        if exist "build\ninja-release\sha1_miner.exe" (
            cd build\ninja-release
            sha1_miner.exe --test
            cd ..\..
        ) else if exist "build\ninja-debug\sha1_miner.exe" (
            cd build\ninja-debug
            sha1_miner.exe --test
            cd ..\..
        ) else if exist "build\vs2022-release\Release\sha1_miner.exe" (
            cd "build\vs2022-release\Release"
            sha1_miner.exe --test
            cd ..\..\..
        ) else if exist "build\vs2022-debug\Debug\sha1_miner.exe" (
            cd "build\vs2022-debug\Debug"
            sha1_miner.exe --test
            cd ..\..\..
        ) else (
            echo No NVIDIA build found! Build first.
        )
    )
    pause
    goto MAIN_MENU
) else if "%test_choice%"=="2" (
    echo.
    set /p "test_cmd=Enter test command: "
    !test_cmd!
    pause
    goto MAIN_MENU
) else if "%test_choice%"=="3" (
    goto MAIN_MENU
) else (
    echo Invalid selection!
    pause
    goto TEST
)

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
if "%clean_choice%"=="1" if exist "build\ninja-release" rmdir /s /q "build\ninja-release" & echo Cleaning build\ninja-release... & echo Done!
if "%clean_choice%"=="1" if not exist "build\ninja-release" echo Directory not found!
if "%clean_choice%"=="2" if exist "build\ninja-debug" rmdir /s /q "build\ninja-debug" & echo Cleaning build\ninja-debug... & echo Done!
if "%clean_choice%"=="2" if not exist "build\ninja-debug" echo Directory not found!
if "%clean_choice%"=="3" if exist "build\vs2022-release" rmdir /s /q "build\vs2022-release" & echo Cleaning build\vs2022-release... & echo Done!
if "%clean_choice%"=="3" if not exist "build\vs2022-release" echo Directory not found!
if "%clean_choice%"=="4" if exist "build\vs2022-debug" rmdir /s /q "build\vs2022-debug" & echo Cleaning build\vs2022-debug... & echo Done!
if "%clean_choice%"=="4" if not exist "build\vs2022-debug" echo Directory not found!
if "%clean_choice%"=="5" if exist "build\hip-release" rmdir /s /q "build\hip-release" & echo Cleaning build\hip-release... & echo Done!
if "%clean_choice%"=="5" if not exist "build\hip-release" echo Directory not found!
if "%clean_choice%"=="6" if exist "build\hip-debug" rmdir /s /q "build\hip-debug" & echo Cleaning build\hip-debug... & echo Done!
if "%clean_choice%"=="6" if not exist "build\hip-debug" echo Directory not found!
if "%clean_choice%"=="7" if exist "build" rmdir /s /q "build" & echo Cleaning all build directories... & echo Done!
if "%clean_choice%"=="8" goto MAIN_MENU
if "%clean_choice%" neq "1" if "%clean_choice%" neq "2" if "%clean_choice%" neq "3" if "%clean_choice%" neq "4" if "%clean_choice%" neq "5" if "%clean_choice%" neq "6" if "%clean_choice%" neq "7" if "%clean_choice%" neq "8" echo Invalid selection!
pause
goto MAIN_MENU

:SETUP_VCPKG
cls
echo =====================================
echo    SETUP VCPKG
echo =====================================
echo.
where git >nul 2>&1
if errorlevel 1 echo ERROR: Git not found! Please install Git and add it to your PATH. & pause & goto MAIN_MENU
where cmake >nul 2>&1
if errorlevel 1 echo ERROR: CMake not found! Please install CMake and add it to your PATH. & pause & goto MAIN_MENU
if exist "vcpkg\vcpkg.exe" (
    echo vcpkg is already installed! Checking for updates...
    cd vcpkg
    git pull
    call bootstrap-vcpkg.bat
    cd ..
) else (
    echo Installing vcpkg...
    git clone https://github.com/Microsoft/vcpkg.git
    if errorlevel 1 echo Failed to clone vcpkg! & pause & goto MAIN_MENU
    cd vcpkg
    call bootstrap-vcpkg.bat
    if errorlevel 1 echo Failed to bootstrap vcpkg! & pause & goto MAIN_MENU
    cd ..
    echo vcpkg installed successfully!
)
echo.
echo Installing required packages for x64-windows-static...
cd vcpkg
.\vcpkg install boost-beast boost-asio boost-system boost-thread boost-program-options boost-date-time boost-regex boost-random boost-chrono boost-atomic openssl zlib nlohmann-json --triplet x64-windows-static
if errorlevel 1 (
    echo ERROR: Package installation failed!
    pause
    goto MAIN_MENU
)
cd ..
echo Package installation complete!
pause
goto MAIN_MENU

:CHECK_AMD_TOOLS
echo Checking AMD/ROCm tools...
if "%ROCM_PATH%"=="" call :AUTO_DETECT_ROCM
if "%ROCM_PATH%"=="" if errorlevel 1 exit /b 1
if not exist "%ROCM_PATH%\bin\hipcc.exe" call :AUTO_DETECT_ROCM
if not exist "%ROCM_PATH%\bin\hipcc.exe" if errorlevel 1 exit /b 1
for %%i in ("%ROCM_PATH%") do set ROCM_VERSION=%%~nxi
echo ROCm Version: %ROCM_VERSION%
set HIP_PATH=%ROCM_PATH%
set HSA_PATH=%ROCM_PATH%
set HIP_PLATFORM=amd
set HIP_RUNTIME=rocclr
set HIP_COMPILER=clang
if "%HIP_ARCHITECTURES%"=="" call :AUTO_DETECT_GPU_ARCH
echo AMD/ROCm tools are functional.
exit /b 0

:AUTO_DETECT_ROCM
echo Searching for ROCm installation...
set ROCM_FOUND=0
if defined ROCM_PATH if exist "%ROCM_PATH%\bin\hipcc.exe" echo Using ROCM_PATH: %ROCM_PATH% & set ROCM_FOUND=1 & exit /b 0
if exist "C:\Program Files\AMD\ROCm" for /d %%v in ("C:\Program Files\AMD\ROCm\*") do if exist "%%v\bin\hipcc.exe" set "ROCM_PATH=%%v" & set ROCM_FOUND=1 & echo Found ROCm at: %%v & exit /b 0
if exist "C:\Program Files\AMD\ROCm\bin\hipcc.exe" set "ROCM_PATH=C:\Program Files\AMD\ROCm" & set ROCM_FOUND=1 & echo Found ROCm at: C:\Program Files\AMD\ROCm & exit /b 0
if exist "C:\ROCm" for /d %%v in ("C:\ROCm\*") do if exist "%%v\bin\hipcc.exe" set "ROCM_PATH=%%v" & set ROCM_FOUND=1 & echo Found ROCm at: %%v & exit /b 0
for /f "tokens=2*" %%a in ('reg query "HKLM\SOFTWARE\AMD\ROCm" /v InstallDir 2^>nul') do if exist "%%b\bin\hipcc.exe" set "ROCM_PATH=%%b" & set ROCM_FOUND=1 & echo Found ROCm via registry at: %%b & exit /b 0
if defined HIP_PATH if exist "%HIP_PATH%\bin\hipcc.exe" set "ROCM_PATH=%HIP_PATH%" & set ROCM_FOUND=1 & echo Found ROCm via HIP_PATH: %HIP_PATH% & exit /b 0
echo ERROR: ROCm not found!
echo Please install ROCm: https://rocm.docs.amd.com/en/latest/deploy/windows/index.html
echo Searched in:
echo - ROCM_PATH, HIP_PATH
echo - C:\Program Files\AMD\ROCm\[version]\
echo - C:\Program Files\AMD\ROCm\
echo - C:\ROCm\[version]\
echo - Registry
pause
exit /b 1

:AUTO_DETECT_GPU_ARCH
echo Auto-detecting GPU architecture...
if exist "%ROCM_PATH%\bin\rocm-smi.exe" (
    "%ROCM_PATH%\bin\rocm-smi.exe" --showproductname >"%TEMP%\gpu_info.txt" 2>nul
    if !ERRORLEVEL!==0 (
        set GPU_ARCHS=
        for /f "tokens=*" %%a in ('findstr /i "GPU" "%TEMP%\gpu_info.txt"') do (
            echo %%a | findstr /i "RX.6[89]00" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx1030,gfx1031,
            echo %%a | findstr /i "RX.7[89]00" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx1100,gfx1101,gfx1102,
            echo %%a | findstr /i "RX.7600" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx1102,
            echo %%a | findstr /i "RX.7700" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx1101,
            echo %%a | findstr /i "RX.7800" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx1101,
            echo %%a | findstr /i "RX.7900" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx1100,
            echo %%a | findstr /i "MI200" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx90a,
            echo %%a | findstr /i "MI250" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx90a,
            echo %%a | findstr /i "MI300" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx940,gfx941,gfx942,
        )
        del "%TEMP%\gpu_info.txt" >nul 2>&1
        if not "!GPU_ARCHS!"=="" (
            set HIP_ARCHITECTURES=!GPU_ARCHS:~0,-1!
            echo Detected GPU architectures: !HIP_ARCHITECTURES!
            exit /b 0
        )
    )
)
if exist "%ROCM_PATH%\bin\hipinfo.exe" (
    "%ROCM_PATH%\bin\hipinfo.exe" >"%TEMP%\gpu_info.txt" 2>nul
    if !ERRORLEVEL!==0 (
        for /f "tokens=2 delims=:" %%a in ('findstr /i "gcnArchName" "%TEMP%\gpu_info.txt"') do (
            set ARCH_NAME=%%a
            set ARCH_NAME=!ARCH_NAME: =!
            if not "!ARCH_NAME!"=="" (
                set HIP_ARCHITECTURES=!ARCH_NAME!
                echo Detected GPU architecture: !HIP_ARCHITECTURES!
                del "%TEMP%\gpu_info.txt" >nul 2>&1
                exit /b 0
            )
        )
        del "%TEMP%\gpu_info.txt" >nul 2>&1
    )
)
echo Could not auto-detect GPU architecture. Using defaults...
set HIP_ARCHITECTURES=gfx1030,gfx1031,gfx1032,gfx1100,gfx1101,gfx1102,gfx1200,gfx1201
echo Default architectures: %HIP_ARCHITECTURES%
exit /b 0

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

:EXIT
echo Exiting build system...
exit /b 0
where cmake >nul 2>&1
if errorlevel 1 echo ERROR: CMake not found! & echo Install CMake and add to PATH. & pause & exit /b 1
where ninja >nul 2>&1
if errorlevel 1 echo WARNING: Ninja not found! & echo Install Ninja for NVIDIA Ninja builds: & echo - winget install Ninja-build.Ninja & echo - https://github.com/ninja-build/ninja/releases
echo Current GPU Backend: %GPU_BACKEND%
pause
goto MAIN_MENU
