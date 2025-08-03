@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
verify >nul

:: Unified build script for SHA1 Miner GPU project
:: Supports NVIDIA/CUDA and AMD/HIP builds on Windows
:: For cross-platform compatibility (Linux, macOS), consider using Python or CMake directly

:: Initialize GPU backend
if "%GPU_BACKEND%"=="" set GPU_BACKEND=NVIDIA

:: Manual ROCm path override - uncomment and modify if auto-detection fails
:: set "ROCM_PATH=C:\Program Files\AMD\ROCm\6.2"

:: Manual CUDA path override - uncomment and modify if auto-detection fails
:: set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"

:MAIN_MENU
cls
echo =====================================
echo    SHA1 Miner Build System (Windows)
echo =====================================
echo    GPU Backend: !GPU_BACKEND!
echo =====================================
echo.
echo 1. Configure Project
echo 2. Build Project
echo 3. Test Project
echo 4. Clean Build Directory
echo 5. Setup vcpkg (if needed)
echo 6. Switch GPU Backend
echo 7. Exit
echo.
set /p choice="Select option (1-7): "

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
echo.
echo Current backend: !GPU_BACKEND!
echo.
echo 1. NVIDIA (CUDA)
echo 2. AMD (HIP/ROCm)
echo 3. Back to Main Menu
echo.
set /p backend_choice="Select GPU backend (1-3): "

if "%backend_choice%"=="1" (
    set GPU_BACKEND=NVIDIA
    echo GPU backend set to NVIDIA/CUDA
) else if "%backend_choice%"=="2" (
    set GPU_BACKEND=AMD
    echo GPU backend set to AMD/HIP
) else if "%backend_choice%"=="3" (
    goto MAIN_MENU
) else (
    echo Invalid selection!
)
pause
goto MAIN_MENU

:CONFIGURE
cls
echo =====================================
echo    Configure Presets
echo =====================================
echo.

:: Setup Visual Studio environment for CMake and Ninja builds
call :SETUP_VS_ENV
if errorlevel 1 (
    echo Configuration aborted due to Visual Studio environment setup failure.
    pause
    goto MAIN_MENU
)

:: Force re-evaluation for debugging
set "GPU_BACKEND=!GPU_BACKEND!"
echo DEBUG: GPU_BACKEND at CONFIGURE start: "!GPU_BACKEND!"

if "!GPU_BACKEND!"=="NVIDIA" (
    goto CONFIGURE_NVIDIA_OPTIONS
) else if "!GPU_BACKEND!"=="AMD" (
    goto CONFIGURE_AMD_OPTIONS
) else (
    echo ERROR: Unknown GPU_BACKEND value: !GPU_BACKEND!
    pause
    goto MAIN_MENU
)

:CONFIGURE_AMD_OPTIONS
    echo Entering AMD/HIP configuration options...
    echo AMD/HIP Build Options:
    echo 1. Windows AMD Release (HIP/ROCm)
    echo 2. Windows AMD Debug (HIP/ROCm)
    echo 3. Back to Main Menu
    echo.
    set /p config_choice="Select configuration (1-3): "

    if "!config_choice!"=="1" (
        set BUILD_DIR=build\hip-release
        set CMAKE_BUILD_TYPE=Release
    ) else if "!config_choice!"=="2" (
        set BUILD_DIR=build\hip-debug
        set CMAKE_BUILD_TYPE=Debug
    ) else if "!config_choice!"=="3" (
        goto MAIN_MENU
    ) else (
        echo Invalid selection!
        pause
        goto CONFIGURE_AMD_OPTIONS
    )

    echo.
    echo Configuring for AMD/HIP build...
    echo.

    :: Check AMD dependencies
    call :CHECK_AMD_TOOLS
    if errorlevel 1 (
        echo Configuration aborted due to missing AMD/ROCm tools.
        pause
        goto MAIN_MENU
    )

    :: Check if vcpkg exists
    if not exist "vcpkg\vcpkg.exe" (
        echo WARNING: vcpkg not found! You may need to run Setup vcpkg first.
        echo.
        set /p continue_anyway="Continue anyway? (y/n): "
        if /i "!continue_anyway!" neq "y" (
            goto MAIN_MENU
        )
    )

    echo Creating AMD/HIP configuration...
    echo Build directory: !BUILD_DIR!

    :: Create build directory if it doesn't exist
    if not exist "!BUILD_DIR!" mkdir "!BUILD_DIR!"

    :: Set vcpkg paths
    set VCPKG_CMAKE=
    if exist "vcpkg\scripts\buildsystems\vcpkg.cmake" (
        set VCPKG_CMAKE=-DCMAKE_TOOLCHAIN_FILE=%CD%\vcpkg\scripts\buildsystems\vcpkg.cmake
        if not defined VCPKG_TARGET_TRIPLET (
            set VCPKG_CMAKE=!VCPKG_CMAKE! -DVCPKG_TARGET_TRIPLET=x64-windows-static
            echo Using static vcpkg triplet for AMD/HIP build (hipcc requirement)
        ) else (
            set VCPKG_CMAKE=!VCPKG_CMAKE! -DVCPKG_TARGET_TRIPLET=!VCPKG_TARGET_TRIPLET!
            echo Using user-specified vcpkg triplet: !VCPKG_TARGET_TRIPLET!
        )
    ) else (
        echo WARNING: vcpkg toolchain not found!
    )

    :: Properly quote ROCM_PATH for CMake
    set "QUOTED_ROCM_PATH=!ROCM_PATH!"
    cmake -B "!BUILD_DIR!" -G "Ninja" -DCMAKE_BUILD_TYPE=!CMAKE_BUILD_TYPE! -DCMAKE_CXX_STANDARD=20 -DUSE_HIP=ON -DROCM_PATH="!QUOTED_ROCM_PATH!" -DHIP_ARCH="!HIP_ARCHITECTURES!" !VCPKG_CMAKE!

    if errorlevel 1 (
        echo.
        echo Configuration failed!
        pause
    ) else (
        echo.
        echo Configuration successful!
        pause
    )
    goto MAIN_MENU

:CONFIGURE_NVIDIA_OPTIONS
    echo Entering NVIDIA/CUDA configuration options...
    echo NVIDIA/CUDA Build Options:
    echo 1. Windows Release (Ninja)
    echo 2. Windows Debug (Ninja)
    echo 3. Windows Release (Visual Studio 2022)
    echo 4. Windows Debug (Visual Studio 2022)
    echo 5. Back to Main Menu
    echo.
    set /p config_choice="Select configuration (1-5): "

    set PRESET=
    set BUILD_DIR=
    if "!config_choice!"=="1" (
        set PRESET=windows-ninja-release
        set BUILD_DIR=build\ninja-release
    ) else if "!config_choice!"=="2" (
        set PRESET=windows-ninja-debug
        set BUILD_DIR=build\ninja-debug
    ) else if "!config_choice!"=="3" (
        set PRESET=windows-vs2022-release
        set BUILD_DIR=build\vs2022-release
    ) else if "!config_choice!"=="4" (
        set PRESET=windows-vs2022-debug
        set BUILD_DIR=build\vs2022-debug
    ) else if "!config_choice!"=="5" (
        goto MAIN_MENU
    ) else (
        echo Invalid selection!
        pause
        goto CONFIGURE_NVIDIA_OPTIONS
    )

    echo.
    echo Configuring with preset: !PRESET!
    echo Build directory: !BUILD_DIR!
    echo.

    :: Check NVIDIA dependencies
    call :CHECK_NVIDIA_TOOLS
    if errorlevel 1 (
        echo Configuration aborted due to missing NVIDIA/CUDA tools.
        pause
        goto MAIN_MENU
    )

    :: Check for Ninja if a Ninja preset is selected
    echo !PRESET! | findstr /i "ninja" >nul
    if not errorlevel 1 (
        where ninja >nul 2>nul
        if errorlevel 1 (
            echo.
            echo ERROR: Ninja build tool not found in PATH!
            echo You selected a Ninja preset, but Ninja is required.
            echo Please install Ninja and add it to your system PATH.
            echo You can download it from https://github.com/ninja-build/ninja/releases
            echo or install via winget: 'winget install Ninja-build.Ninja'
            echo.
            pause
            goto MAIN_MENU
        ) else (
            echo [✓] Ninja build tool found.
        )
    )

    :: Auto-detect CUDA architectures if not set
    if "!CUDA_ARCHITECTURES!"=="" (
        call :AUTO_DETECT_CUDA_ARCH
    )
    echo CUDA Architectures: !CUDA_ARCHITECTURES!

    :: Set vcpkg paths for NVIDIA (dynamic linking preferred)
    set VCPKG_CMAKE=
    if exist "vcpkg\scripts\buildsystems\vcpkg.cmake" (
        set VCPKG_CMAKE=-DCMAKE_TOOLCHAIN_FILE=%CD%\vcpkg\scripts\buildsystems\vcpkg.cmake
        if not defined VCPKG_TARGET_TRIPLET (
            set VCPKG_CMAKE=!VCPKG_CMAKE! -DVCPKG_TARGET_TRIPLET=x64-windows
            echo Using dynamic vcpkg triplet for NVIDIA/CUDA build
        ) else (
            set VCPKG_CMAKE=!VCPKG_CMAKE! -DVCPKG_TARGET_TRIPLET=!VCPKG_TARGET_TRIPLET!
            echo Using user-specified vcpkg triplet: !VCPKG_TARGET_TRIPLET!
        )
    ) else (
        echo WARNING: vcpkg toolchain not found!
    )

    :: Run CMake configure for NVIDIA with CUDA architectures
    cmake --preset !PRESET! -DCUDA_ARCHITECTURES="!CUDA_ARCHITECTURES!" !VCPKG_CMAKE!

    if errorlevel 1 (
        echo.
        echo Configuration failed!
        pause
    ) else (
        echo.
        echo Configuration successful!
        pause
    )
    goto MAIN_MENU

:BUILD
cls
echo ==============================
echo    Build Project
==============================
echo.

:: Setup Visual Studio environment for builds
call :SETUP_VS_ENV
if errorlevel 1 (
    echo Build aborted due to Visual Studio environment setup failure.
    pause
    goto MAIN_MENU
)

:: Force re-evaluation for debugging
set "GPU_BACKEND=!GPU_BACKEND!"
echo DEBUG: GPU_BACKEND at BUILD start: "!GPU_BACKEND!"

if "!GPU_BACKEND!"=="NVIDIA" (
    goto BUILD_NVIDIA_OPTIONS
) else if "!GPU_BACKEND!"=="AMD" (
    goto BUILD_AMD_OPTIONS
) else (
    echo ERROR: Unknown GPU_BACKEND value: !GPU_BACKEND!
    pause
    goto MAIN_MENU
)

:BUILD_AMD_OPTIONS
    echo Entering AMD/HIP build options...
    echo AMD/HIP Build Options:
    echo 1. Windows AMD Release
    echo 2. Windows AMD Debug
    echo 3. Back to Main Menu
    echo.
    set /p build_choice="Select build configuration (1-3): "

    if "!build_choice!"=="1" (
        set BUILD_DIR=build\hip-release
        set BUILD_CONFIG=Release
    ) else if "!build_choice!"=="2" (
        set BUILD_DIR=build\hip-debug
        set BUILD_CONFIG=Debug
    ) else if "!build_choice!"=="3" (
        goto MAIN_MENU
    ) else (
        echo Invalid selection!
        pause
        goto BUILD_AMD_OPTIONS
    )

    echo.
    echo Building AMD/HIP configuration...
    echo Build directory: !BUILD_DIR!

    if not exist "!BUILD_DIR!" (
        echo.
        echo ERROR: Build directory !BUILD_DIR! does not exist!
        echo Please configure the project first - Option 1 in main menu.
        echo.
        echo Steps:
        echo 1. Go back to main menu
        echo 2. Select "1. Configure Project"
        echo 3. Choose AMD configuration - Release or Debug
        echo 4. Then come back here to build
        echo.
        pause
        goto MAIN_MENU
    )

    if not exist "!BUILD_DIR!\CMakeCache.txt" (
        echo.
        echo ERROR: Project not configured in !BUILD_DIR!
        echo Please configure the project first.
        pause
        goto MAIN_MENU
    )

    :: Use number of logical processors for parallel build
    :: Use system variable NUMBER_OF_PROCESSORS instead of wmic
    set "NUM_CORES=%NUMBER_OF_PROCESSORS%"
    if not defined NUM_CORES set NUM_CORES=4

    echo Building with !NUM_CORES! parallel jobs...
    cmake --build "!BUILD_DIR!" --config !BUILD_CONFIG! --verbose -j!NUM_CORES!

    if errorlevel 1 (
        echo.
        echo Build failed!
        pause
    ) else (
        echo.
        echo Build successful!
        pause
    )
    goto MAIN_MENU

:BUILD_NVIDIA_OPTIONS
    echo Entering NVIDIA/CUDA build options...
    echo NVIDIA/CUDA Build Options:
    echo 1. Windows Release (Ninja)
    echo 2. Windows Debug (Ninja)
    echo 3. Windows Release (Visual Studio)
    echo 4. Windows Debug (Visual Studio)
    echo 5. Back to Main Menu
    echo.
    set /p build_choice="Select build configuration (1-5): "

    set BUILD_PRESET=
    set BUILD_DIR=
    if "!build_choice!"=="1" (
        set BUILD_PRESET=windows-ninja-release
        set BUILD_DIR=build\ninja-release
    ) else if "!build_choice!"=="2" (
        set BUILD_PRESET=windows-ninja-debug
        set BUILD_DIR=build\ninja-debug
    ) else if "!build_choice!"=="3" (
        set BUILD_PRESET=windows-vs2022-release
        set BUILD_DIR=build\vs2022-release
    ) else if "!build_choice!"=="4" (
        set BUILD_PRESET=windows-vs2022-debug
        set BUILD_DIR=build\vs2022-debug
    ) else if "!build_choice!"=="5" (
        goto MAIN_MENU
    ) else (
        echo Invalid selection!
        pause
        goto BUILD_NVIDIA_OPTIONS
    )

    echo.
    echo Building with preset: !BUILD_PRESET!
    echo Build directory: !BUILD_DIR!
    echo.

    :: Check for Ninja if a Ninja preset is selected
    echo !BUILD_PRESET! | findstr /i "ninja" >nul
    if not errorlevel 1 (
        where ninja >nul 2>nul
        if errorlevel 1 (
            echo.
            echo ERROR: Ninja build tool not found in PATH!
            echo You selected a Ninja preset, but Ninja is required.
            echo Please install Ninja and add it to your system PATH.
            echo You can download it from https://github.com/ninja-build/ninja/releases
            echo or install via winget: 'winget install Ninja-build.Ninja'
            echo.
            pause
            goto MAIN_MENU
        ) else (
            echo [✓] Ninja build tool found.
        )
    )

    if not exist "!BUILD_DIR!" (
        echo.
        echo ERROR: Build directory !BUILD_DIR! does not exist!
        echo Please configure the project first - Option 1 in main menu.
        echo.
        echo Steps:
        echo 1. Go back to main menu
        echo 2. Select "1. Configure Project"
        echo 3. Choose NVIDIA configuration
        echo 4. Then come back here to build
        echo.
        pause
        goto MAIN_MENU
    )

    :: Use number of logical processors for parallel build
    set "NUM_CORES=%NUMBER_OF_PROCESSORS%"
    if not defined NUM_CORES set NUM_CORES=4

    echo Building with !NUM_CORES! parallel jobs...
    cmake --build --preset !BUILD_PRESET! -j!NUM_CORES!

    if errorlevel 1 (
        echo.
        echo Build failed!
        pause
    ) else (
        echo.
        echo Build successful!
        pause
    )
    goto MAIN_MENU

:TEST
cls
echo =====================================
echo    Test Presets
=====================================
echo.
echo 1. Test current configuration
echo 2. Run custom test command
echo 3. Back to Main Menu
echo.
set /p test_choice="Select test option (1-3): "

if "%test_choice%"=="1" (
    echo.
    if "!GPU_BACKEND!"=="AMD" (
        echo Running tests for AMD/HIP build...
        set TEST_EXECUTABLE=
        if exist "build\hip-release\sha1_miner.exe" (
            set TEST_EXECUTABLE=build\hip-release\sha1_miner.exe
        ) else if exist "build\hip-debug\sha1_miner.exe" (
            set TEST_EXECUTABLE=build\hip-debug\sha1_miner.exe
        )

        if defined TEST_EXECUTABLE (
            echo Running: "!TEST_EXECUTABLE!" --test
            "!TEST_EXECUTABLE!" --test
        ) else (
            echo No AMD build found! Build first.
        )
    ) else (
        echo Running tests for NVIDIA/CUDA build...
        set TEST_EXECUTABLE=
        if exist "build\ninja-release\sha1_miner.exe" (
            set TEST_EXECUTABLE=build\ninja-release\sha1_miner.exe
        ) else if exist "build\ninja-debug\sha1_miner.exe" (
            set TEST_EXECUTABLE=build\ninja-debug\sha1_miner.exe
        ) else if exist "build\vs2022-release\sha1_miner.exe" (
            set TEST_EXECUTABLE=build\vs2022-release\sha1_miner.exe
        ) else if exist "build\vs2022-debug\sha1_miner.exe" (
            set TEST_EXECUTABLE=build\vs2022-debug\sha1_miner.exe
        )

        if defined TEST_EXECUTABLE (
            echo Running: "!TEST_EXECUTABLE!" --test
            "!TEST_EXECUTABLE!" --test
        ) else (
            echo No NVIDIA build found! Build and configure first.
        )
    )
    pause
    goto MAIN_MENU
) else if "%test_choice%"=="2" (
    echo.
    set /p test_cmd="Enter test command: "
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
echo    Clean Build Directories
=====================================
echo.
echo 1. Clean Release build (NVIDIA/CUDA - Ninja)
echo 2. Clean Debug build (NVIDIA/CUDA - Ninja)
echo 3. Clean VS2022 Release build (NVIDIA/CUDA)
echo 4. Clean VS2022 Debug build (NVIDIA/CUDA)
echo 5. Clean AMD/HIP Release build
echo 6. Clean AMD/HIP Debug build
echo 7. Clean all builds
echo 8. Back to Main Menu
echo.
set /p clean_choice="Select directory to clean (1-8): "

if "%clean_choice%"=="1" (
    if exist "build\ninja-release" (
        echo Cleaning build\ninja-release...
        rmdir /s /q "build\ninja-release"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "%clean_choice%"=="2" (
    if exist "build\ninja-debug" (
        echo Cleaning build\ninja-debug...
        rmdir /s /q "build\ninja-debug"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "%clean_choice%"=="3" (
    if exist "build\vs2022-release" (
        echo Cleaning build\vs2022-release...
        rmdir /s /q "build\vs2022-release"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "%clean_choice%"=="4" (
    if exist "build\vs2022-debug" (
        echo Cleaning build\vs2022-debug...
        rmdir /s /q "build\vs2022-debug"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "%clean_choice%"=="5" (
    if exist "build\hip-release" (
        echo Cleaning build\hip-release...
        rmdir /s /q "build\hip-release"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "%clean_choice%"=="6" (
    if exist "build\hip-debug" (
        echo Cleaning build\hip-debug...
        rmdir /s /q "build\hip-debug"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "%clean_choice%"=="7" (
    echo Cleaning all build directories...
    if exist "build" rmdir /s /q "build"
    echo Done!
) else if "%clean_choice%"=="8" (
    goto MAIN_MENU
) else (
    echo Invalid selection!
)
pause
goto MAIN_MENU

:SETUP_VCPKG
cls
echo =====================================
echo    Setup vcpkg
=====================================
echo.

:: Check for Git
where git >nul 2>nul
if errorlevel 1 (
    echo ERROR: Git not found in PATH!
    echo Please install Git from https://git-scm.com/ and add it to your PATH.
    pause
    goto MAIN_MENU
)

if exist "vcpkg\vcpkg.exe" (
    echo vcpkg is already installed!
    echo.
    set /p update_choice="Do you want to update vcpkg? (y/n): "
    if /i "!update_choice!"=="y" (
        pushd vcpkg
        git pull
        call bootstrap-vcpkg.bat
        popd
    )
) else (
    echo vcpkg not found. Installing...
    echo.

    :: Clone vcpkg
    git clone https://github.com/Microsoft/vcpkg.git

    :: Bootstrap vcpkg
    pushd vcpkg
    call bootstrap-vcpkg.bat
    popd

    echo.
    echo vcpkg installed successfully!
)

echo.
echo Installing required packages...
echo This may take a while...
echo.

pushd vcpkg
:: Install both static and dynamic libraries for flexibility
.\vcpkg install boost-beast boost-asio boost-system boost-thread boost-program-options boost-date-time boost-regex boost-random openssl curl zlib nlohmann-json --triplet x64-windows --triplet x64-windows-static
popd

echo.
echo Package installation complete!
pause
goto MAIN_MENU

:CHECK_NVIDIA_TOOLS
:: Check for CUDA
set CUDA_FOUND=0
set CUDA_PATH_DETECTED=

:: Check environment variable first
if defined CUDA_PATH (
    if exist "%CUDA_PATH%\bin\nvcc.exe" (
        set CUDA_FOUND=1
        set CUDA_PATH_DETECTED=%CUDA_PATH%
    )
)

:: Try common CUDA installation paths if not found via env var
if %CUDA_FOUND%==0 (
    for /d %%i in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
        if exist "%%i\bin\nvcc.exe" (
            set CUDA_FOUND=1
            set CUDA_PATH_DETECTED=%%i
            goto :CUDA_FOUND_LABEL
        )
    )
)

:CUDA_FOUND_LABEL
if %CUDA_FOUND%==1 (
    echo [✓] CUDA Toolkit found at: !CUDA_PATH_DETECTED!
    set "CUDA_PATH=!CUDA_PATH_DETECTED!"
    "!CUDA_PATH!\bin\nvcc.exe" --version >nul 2>&1
    if errorlevel 0 (
        echo [✓] nvcc is functional.
        exit /b 0
    ) else (
        echo [!] nvcc found but not functional.
        exit /b 1
    )
) else (
    echo WARNING: CUDA Toolkit not found!
    echo CUDA builds may fail.
    echo Please ensure CUDA Toolkit is installed and CUDA_PATH is set.
    echo Example: set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2
    exit /b 1
)

:CHECK_AMD_TOOLS
:: Set default ROCm path if not set
if "!ROCM_PATH!"=="" (
    call :AUTO_DETECT_ROCM
    if errorlevel 1 exit /b 1
)

:: Validate ROCm installation
if not exist "!ROCM_PATH!\bin\hipcc.bin.exe" (
    if not exist "!ROCM_PATH!\bin\hipcc.exe" (
        echo ERROR: hipcc not found in !ROCM_PATH!\bin\
        echo Attempting to re-detect ROCm installation...
        call :AUTO_DETECT_ROCM
        if errorlevel 1 exit /b 1
    )
)

:: Extract ROCm version from path
for %%i in ("!ROCM_PATH!") do set ROCM_VERSION=%%~nxi
echo ROCm Version: !ROCM_VERSION!

:: Set HIP environment variables
set HIP_PATH=!ROCM_PATH!
set HSA_PATH=!ROCM_PATH!
set HIP_PLATFORM=amd
set HIP_RUNTIME=rocclr
set HIP_COMPILER=clang

:: Auto-detect GPU architecture if not set
if "!HIP_ARCHITECTURES!"=="" (
    call :AUTO_DETECT_GPU_ARCH
)

:: Check ROCm components
call :CHECK_ROCM_COMPONENTS
if errorlevel 1 (
    echo ROCM components check failed.
    exit /b 1
)

echo.
echo ROCm Configuration Summary:
echo    Installation: !ROCM_PATH!
echo    Version: !ROCM_VERSION!
echo    HIP Architectures: !HIP_ARCHITECTURES!
echo.
exit /b 0

:AUTO_DETECT_ROCM
echo Searching for ROCm installation...
set ROCM_FOUND=0
set ROCM_PATH=

:: Check ROCM_PATH environment variable
if defined ROCM_PATH (
    if exist "!ROCM_PATH!\bin\hipcc.bin.exe" (
        echo Using ROCM_PATH from environment: !ROCM_PATH!
        set ROCM_FOUND=1
        exit /b 0
    )
    if exist "!ROCM_PATH!\bin\hipcc.exe" (
        echo Using ROCM_PATH from environment: !ROCM_PATH!
        set ROCM_FOUND=1
        exit /b 0
    )
)

:: Check Program Files for versioned installations
if exist "%ProgramFiles%\AMD\ROCm" (
    for /d %%v in ("%ProgramFiles%\AMD\ROCm\*") do (
        if exist "%%v\bin\hipcc.bin.exe" (
            set "ROCM_PATH=%%v"
            set ROCM_FOUND=1
            echo Found ROCm at: %%v
            exit /b 0
        )
        if exist "%%v\bin\hipcc.exe" (
            set "ROCM_PATH=%%v"
            set ROCM_FOUND=1
            echo Found ROCm at: %%v
            exit /b 0
        )
    )
)

:: Check for direct installation
if exist "%ProgramFiles%\AMD\ROCm\bin\hipcc.bin.exe" (
    set "ROCM_PATH=%ProgramFiles%\AMD\ROCm"
    set ROCM_FOUND=1
    echo Found ROCm at: %ProgramFiles%\AMD\ROCm
    exit /b 0
)
if exist "%ProgramFiles%\AMD\ROCm\bin\hipcc.exe" (
    set "ROCM_PATH=%ProgramFiles%\AMD\ROCm"
    set ROCM_FOUND=1
    echo Found ROCm at: %ProgramFiles%\AMD\ROCm
    exit /b 0
)

:: Check C:\ROCm
if exist "C:\ROCm" (
    for /d %%v in ("C:\ROCm\*") do (
        if exist "%%v\bin\hipcc.bin.exe" (
            set "ROCM_PATH=%%v"
            set ROCM_FOUND=1
            echo Found ROCm at: %%v
            exit /b 0
        )
        if exist "%%v\bin\hipcc.exe" (
            set "ROCM_PATH=%%v"
            set ROCM_FOUND=1
            echo Found ROCm at: %%v
            exit /b 0
        )
    )
)

:: Check registry
for /f "tokens=2*" %%a in ('reg query "HKLM\SOFTWARE\AMD\ROCm" /v InstallDir 2^>nul') do (
    if exist "%%b\bin\hipcc.bin.exe" (
        set "ROCM_PATH=%%b"
        set ROCM_FOUND=1
        echo Found ROCm via registry at: %%b
        exit /b 0
    )
    if exist "%%b\bin\hipcc.exe" (
        set "ROCM_PATH=%%b"
        set ROCM_FOUND=1
        echo Found ROCm via registry at: %%b
        exit /b 0
    )
)

:: Check HIP_PATH
if defined HIP_PATH (
    if exist "!HIP_PATH!\bin\hipcc.bin.exe" (
        set "ROCM_PATH=!HIP_PATH!"
        set ROCM_FOUND=1
        echo Found ROCm via HIP_PATH: !HIP_PATH!
        exit /b 0
    )
    if exist "!HIP_PATH!\bin\hipcc.exe" (
        set "ROCM_PATH=!HIP_PATH!"
        set ROCM_FOUND=1
        echo Found ROCm via HIP_PATH: !HIP_PATH!
        exit /b 0
    )
)

:: ROCm not found
echo.
echo ERROR: ROCm not found!
echo Please either:
echo 1. Install ROCm from: https://www.amd.com/en/developer/rocm-software.html
echo 2. Set ROCM_PATH environment variable
echo    Example: set ROCM_PATH=C:\Program Files\AMD\ROCm\6.2
echo Searched in:
echo    - %%ProgramFiles%%\AMD\ROCm\[version]\
echo    - %%ProgramFiles%%\AMD\ROCm\
echo    - C:\ROCm\[version]\
echo    - Windows Registry (HKLM\SOFTWARE\AMD\ROCm)
echo    - Environment variables (ROCM_PATH, HIP_PATH)
echo.
pause
exit /b 1

:AUTO_DETECT_GPU_ARCH
echo Auto-detecting GPU architecture...
set HIP_ARCHITECTURES=

:: Try rocm-smi
if exist "!ROCM_PATH!\bin\rocm-smi.exe" (
    "!ROCM_PATH!\bin\rocm-smi.exe" --showproductname >"%TEMP%\gpu_info.txt" 2>nul
    if !errorlevel!==0 (
        set "GPU_ARCHS_TEMP="
        for /f "tokens=*" %%a in ('type "%TEMP%\gpu_info.txt"') do (
            echo %%a | findstr /i "RX.6[89]00" >nul && set GPU_ARCHS_TEMP=!GPU_ARCHS_TEMP!gfx1030,gfx1031,
            echo %%a | findstr /i "RX.7[6-9]00" >nul && set GPU_ARCHS_TEMP=!GPU_ARCHS_TEMP!gfx1100,gfx1101,gfx1102,
            echo %%a | findstr /i "MI2[05]0" >nul && set GPU_ARCHS_TEMP=!GPU_ARCHS_TEMP!gfx90a,
            echo %%a | findstr /i "MI300" >nul && set GPU_ARCHS_TEMP=!GPU_ARCHS_TEMP!gfx940,gfx941,gfx942,
        )
        del "%TEMP%\gpu_info.txt" >nul 2>&1
        if not "!GPU_ARCHS_TEMP!"=="" (
            call :REMOVE_DUPLICATES "!GPU_ARCHS_TEMP!" HIP_ARCHITECTURES
            echo Detected GPU architectures: !HIP_ARCHITECTURES!
            exit /b 0
        )
    )
)

:: Try hipinfo
if exist "!ROCM_PATH!\bin\hipinfo.exe" (
    "!ROCM_PATH!\bin\hipinfo.exe" >"%TEMP%\hip_info.txt" 2>nul
    if !errorlevel!==0 (
        for /f "tokens=2 delims=:" %%a in ('findstr /i "gcnArchName" "%TEMP%\hip_info.txt"') do (
            set ARCH_NAME=%%a
            set ARCH_NAME=!ARCH_NAME: =!
            if not "!ARCH_NAME!"=="" (
                set HIP_ARCHITECTURES=!ARCH_NAME!
                echo Detected GPU architecture: !HIP_ARCHITECTURES!
                del "%TEMP%\hip_info.txt" >nul 2>&1
                exit /b 0
            )
        )
        del "%TEMP%\hip_info.txt" >nul 2>&1
    )
)

:: Fallback to default architectures
echo Could not auto-detect GPU architecture. Using defaults for RDNA2/RDNA3/RDNA4...
set HIP_ARCHITECTURES=gfx1030,gfx1031,gfx1032,gfx1100,gfx1101,gfx1102,gfx1200,gfx1201
echo Default architectures: !HIP_ARCHITECTURES!
exit /b 0

:AUTO_DETECT_CUDA_ARCH
echo Auto-detecting CUDA architecture...
set CUDA_ARCHITECTURES=

:: Try nvidia-smi to detect GPU
where nvidia-smi >nul 2>nul
if !errorlevel!==0 (
    nvidia-smi --query-gpu=compute_cap --format=csv >"%TEMP%\cuda_info.txt" 2>nul
    if !errorlevel!==0 (
        set "CUDA_ARCHS_TEMP="
        for /f "skip=1 tokens=2 delims=," %%a in ('type "%TEMP%\cuda_info.txt"') do (
            set "CAP=%%a"
            set "CAP=!CAP:.=!"
            set CUDA_ARCHS_TEMP=!CUDA_ARCHS_TEMP!!CAP!;
        )
        del "%TEMP%\cuda_info.txt" >nul 2>&1
        if not "!CUDA_ARCHS_TEMP!"=="" (
            call :REMOVE_DUPLICATES "!CUDA_ARCHS_TEMP!" CUDA_ARCHITECTURES
            echo Detected CUDA architectures: !CUDA_ARCHITECTURES!
            exit /b 0
        )
    )
)

:: Fallback to common CUDA architectures
echo Could not auto-detect CUDA architecture. Using defaults...
set CUDA_ARCHITECTURES=60;70;75;80;86;89;90
echo Default CUDA architectures: !CUDA_ARCHITECTURES!
exit /b 0

:REMOVE_DUPLICATES
set "INPUT_LIST=%~1"
set "OUTPUT_VAR_NAME=%~2"
set "UNIQUE_ITEMS="
for %%i in (%INPUT_LIST%) do (
    echo !UNIQUE_ITEMS! | findstr /i "\<%%i\>" >nul || set UNIQUE_ITEMS=!UNIQUE_ITEMS!%%i;
)
set "%OUTPUT_VAR_NAME%=!UNIQUE_ITEMS:~0,-1!"
exit /b 0

:CHECK_ROCM_COMPONENTS
echo.
echo Checking ROCm components:
set ROCM_COMPONENTS_OK=0

set HIPCC_PATH=
if exist "!ROCM_PATH!\bin\hipcc.bin.exe" (
    set HIPCC_PATH="!ROCM_PATH!\bin\hipcc.bin.exe"
) else if exist "!ROCM_PATH!\bin\hipcc.exe" (
    set HIPCC_PATH="!ROCM_PATH!\bin\hipcc.exe"
)

if defined HIPCC_PATH (
    echo [✓] hipcc found (!HIPCC_PATH!)
    !HIPCC_PATH! --version >nul 2>&1
    if !errorlevel!==0 (
        echo [✓] hipcc is functional
        set ROCM_COMPONENTS_OK=1
    ) else (
        echo [!] hipcc found but not functional
    )
) else (
    echo [✗] hipcc NOT found
)

if exist "!ROCM_PATH!\bin\hipconfig.exe" (
    echo [✓] hipconfig found
) else (
    echo [✗] hipconfig NOT found
)

if exist "!ROCM_PATH!\lib\cmake\hip" (
    echo [✓] HIP CMake modules found
) else (
    echo [✗] HIP CMake modules NOT found
)

if exist "!ROCM_PATH!\bin\amdhip64.dll" (
    echo [✓] HIP runtime library found
) else (
    echo [✗] HIP runtime library NOT found
)

if exist "!ROCM_PATH!\bin\rocm-smi.exe" (
    echo [✓] rocm-smi found
) else (
    echo [!] rocm-smi NOT found (optional)
)

if !ROCM_COMPONENTS_OK!==1 (
    exit /b 0
) else (
    exit /b 1
)

:EXIT
echo.
echo Exiting build system...
exit /b 0

:SETUP_VS_ENV
:: Check if VS environment is already set up
if defined VSCMD_VER (
    echo Visual Studio environment already configured
    exit /b 0
)

echo Setting up Visual Studio environment...

:: Try to find Visual Studio 2022
set VCVARS_BAT=
set "VS_COMMUNITY_PATH=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "VS_PROFESSIONAL_PATH=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
set "VS_ENTERPRISE_PATH=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set "VS_BUILDTOOLS_PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if exist "!VS_COMMUNITY_PATH!" (
    set VCVARS_BAT="!VS_COMMUNITY_PATH!"
) else if exist "!VS_PROFESSIONAL_PATH!" (
    set VCVARS_BAT="!VS_PROFESSIONAL_PATH!"
) else if exist "!VS_ENTERPRISE_PATH!" (
    set VCVARS_BAT="!VS_ENTERPRISE_PATH!"
) else if exist "!VS_BUILDTOOLS_PATH!" (
    set VCVARS_BAT="!VS_BUILDTOOLS_PATH!"
)

:: Fallback: Use vswhere
if not defined VCVARS_BAT (
    where vswhere >nul 2>nul
    if not errorlevel 1 (
        for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
            if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" (
                set VCVARS_BAT="%%i\VC\Auxiliary\Build\vcvars64.bat"
                goto :VCVARS_FOUND
            )
        )
    )
)

:VCVARS_FOUND
if defined VCVARS_BAT (
    echo Calling: !VCVARS_BAT!
    call !VCVARS_BAT!
    if errorlevel 1 (
        echo ERROR: Failed to initialize Visual Studio environment from: !VCVARS_BAT!
        pause
        exit /b 1
    )
    echo Visual Studio environment loaded.
    exit /b 0
) else (
    echo ERROR: Could not find Visual Studio 2022 installation!
    echo Please ensure Visual Studio 2022 with C++ development tools is installed.
    echo Alternatively, run this script from a "Developer Command Prompt for VS 2022".
    pause
    exit /b 1
)

:: Check for required tools at startup
where cmake >nul 2>nul
if errorlevel 1 (
    echo ERROR: CMake not found in PATH!
    echo Please install CMake from https://cmake.org/download/ and add it to your PATH.
    pause
    exit /b 1
)

echo Current GPU Backend: !GPU_BACKEND!
echo.
pause
goto MAIN_MENU
