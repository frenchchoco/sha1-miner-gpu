@echo off
setlocal enabledelayedexpansion

:: Build script for SHA1 Miner GPU project
:: Supports AMD/HIP builds

:: Initialize GPU backend
if "!GPU_BACKEND!"=="" set GPU_BACKEND=AMD

:MAIN_MENU
cls
echo =====================================
echo   SHA1 Miner Build System
echo =====================================
echo   GPU Backend: %GPU_BACKEND%
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

if "!choice!"=="1" goto CONFIGURE
if "!choice!"=="2" goto BUILD
if "!choice!"=="3" goto TEST
if "!choice!"=="4" goto CLEAN
if "!choice!"=="5" goto SETUP_VCPKG
if "!choice!"=="6" goto SWITCH_BACKEND
if "!choice!"=="7" goto EXIT
goto MAIN_MENU

:SWITCH_BACKEND
cls
echo =====================================
echo   Select GPU Backend
echo =====================================
echo.
echo Current backend: %GPU_BACKEND%
echo.
echo 1. NVIDIA (CUDA)
echo 2. AMD (HIP/ROCm)
echo 3. Back to Main Menu
echo.
set /p backend_choice="Select GPU backend (1-3): "

if "!backend_choice!"=="1" (
    set GPU_BACKEND=NVIDIA
    echo GPU backend set to NVIDIA/CUDA
) else if "!backend_choice!"=="2" (
    set GPU_BACKEND=AMD
    echo GPU backend set to AMD/HIP
) else if "!backend_choice!"=="3" (
    goto MAIN_MENU
) else (
    echo Invalid selection!
)
pause
goto MAIN_MENU

:CONFIGURE
cls
echo =====================================
echo   Configure Presets
echo =====================================
echo.

:: Setup Visual Studio environment for Ninja builds
call :SETUP_VS_ENV

if "%GPU_BACKEND%"=="AMD" (
    echo AMD/HIP Build Options:
    echo 1. Windows AMD Release (HIP/ROCm)
    echo 2. Windows AMD Debug (HIP/ROCm)
    echo 3. Back to Main Menu
    echo.
    set /p config_choice="Select configuration (1-3): "

    if "!config_choice!"=="1" (
        set PRESET=windows-hip-release
        set BUILD_DIR=build\hip-release
        set CMAKE_BUILD_TYPE=Release
    ) else if "!config_choice!"=="2" (
        set PRESET=windows-hip-debug
        set BUILD_DIR=build\hip-debug
        set CMAKE_BUILD_TYPE=Debug
    ) else if "!config_choice!"=="3" (
        goto MAIN_MENU
    ) else (
        echo Invalid selection!
        pause
        goto CONFIGURE
    )

    echo.
    echo Configuring with preset: %PRESET%
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

    :: Create AMD/HIP configuration
    echo Creating AMD/HIP configuration...
    echo Build directory: %BUILD_DIR%

    :: Create build directory if it doesn't exist
    if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

    :: Set vcpkg paths
    if exist "vcpkg\scripts\buildsystems\vcpkg.cmake" (
        set VCPKG_CMAKE=-DCMAKE_TOOLCHAIN_FILE=%CD%\vcpkg\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
    ) else (
        echo WARNING: vcpkg toolchain not found!
        set VCPKG_CMAKE=
    )

    cmake -B %BUILD_DIR% -G "Ninja" -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE% -DCMAKE_CXX_STANDARD=20 -DUSE_HIP=ON -DROCM_PATH="%ROCM_PATH%" %VCPKG_CMAKE%
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
) else (
    echo NVIDIA/CUDA Build Options:
    echo 1. Windows Release (Ninja + CUDA 12.9)
    echo 2. Windows Debug (Ninja + CUDA 12.9)
    echo 3. Windows Release (Visual Studio 2022)
    echo 4. Windows Debug (Visual Studio 2022)
    echo 5. Back to Main Menu
    echo.
    set /p config_choice="Select configuration (1-5): "

    if "!config_choice!"=="1" (
        set PRESET=windows-ninja-release
        set BUILD_DIR=build\release
    ) else if "!config_choice!"=="2" (
        set PRESET=windows-ninja-debug
        set BUILD_DIR=build\debug
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
        goto CONFIGURE
    )

    echo.
    echo Configuring with preset: %PRESET%
    echo.

    :: Check NVIDIA dependencies
    call :CHECK_NVIDIA_TOOLS

    :: Run CMake configure for NVIDIA
    cmake --preset %PRESET%
)

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
echo   Build Presets
echo ==============================
echo.

:: Setup Visual Studio environment for builds
call :SETUP_VS_ENV

if "%GPU_BACKEND%"=="AMD" (
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
        goto BUILD
    )

    echo.
    echo Building AMD/HIP configuration...
    echo Build directory: %BUILD_DIR%

    if not exist "%BUILD_DIR%" (
        echo.
        echo ERROR: Build directory %BUILD_DIR% does not exist!
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

    if not exist "%BUILD_DIR%\CMakeCache.txt" (
        echo.
        echo ERROR: Project not configured in %BUILD_DIR%
        echo Please configure the project first.
        pause
        goto MAIN_MENU
    )

    cmake --build %BUILD_DIR% -j 12
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
) else (
    echo NVIDIA/CUDA Build Options:
    echo 1. Windows Release (Ninja)
    echo 2. Windows Debug (Ninja)
    echo 3. Windows Release (Visual Studio)
    echo 4. Windows Debug (Visual Studio)
    echo 5. Back to Main Menu
    echo.
    set /p build_choice="Select build configuration (1-5): "

    if "!build_choice!"=="1" (
        set BUILD_PRESET=windows-release
    ) else if "!build_choice!"=="2" (
        set BUILD_PRESET=windows-debug
    ) else if "!build_choice!"=="3" (
        set BUILD_PRESET=windows-release-vs
    ) else if "!build_choice!"=="4" (
        set BUILD_PRESET=windows-debug-vs
    ) else if "!build_choice!"=="5" (
        goto MAIN_MENU
    ) else (
        echo Invalid selection!
        pause
        goto BUILD
    )

    echo.
    echo Building with preset: %BUILD_PRESET%
    cmake --build --preset %BUILD_PRESET%
)

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
echo   Test Presets
echo =====================================
echo.
echo 1. Test current configuration
echo 2. Run custom test command
echo 3. Back to Main Menu
echo.
set /p test_choice="Select test option (1-3): "

if "!test_choice!"=="1" (
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
        ctest --preset test-windows-release
    )
    pause
    goto MAIN_MENU
) else if "!test_choice!"=="2" (
    echo.
    set /p test_cmd="Enter test command: "
    !test_cmd!
    pause
    goto MAIN_MENU
) else if "!test_choice!"=="3" (
    goto MAIN_MENU
) else (
    echo Invalid selection!
    pause
    goto TEST
)

:CLEAN
cls
echo =====================================
echo   Clean Build Directories
echo =====================================
echo.
echo 1. Clean Release build
echo 2. Clean Debug build
echo 3. Clean VS2022 Release build
echo 4. Clean VS2022 Debug build
echo 5. Clean AMD/HIP Release build
echo 6. Clean AMD/HIP Debug build
echo 7. Clean all builds
echo 8. Back to Main Menu
echo.
set /p clean_choice="Select directory to clean (1-8): "

if "!clean_choice!"=="1" (
    if exist "build\release" (
        echo Cleaning build\release...
        rmdir /s /q "build\release"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "!clean_choice!"=="2" (
    if exist "build\debug" (
        echo Cleaning build\debug...
        rmdir /s /q "build\debug"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "!clean_choice!"=="3" (
    if exist "build\vs2022-release" (
        echo Cleaning build\vs2022-release...
        rmdir /s /q "build\vs2022-release"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "!clean_choice!"=="4" (
    if exist "build\vs2022-debug" (
        echo Cleaning build\vs2022-debug...
        rmdir /s /q "build\vs2022-debug"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "!clean_choice!"=="5" (
    if exist "build\hip-release" (
        echo Cleaning build\hip-release...
        rmdir /s /q "build\hip-release"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "!clean_choice!"=="6" (
    if exist "build\hip-debug" (
        echo Cleaning build\hip-debug...
        rmdir /s /q "build\hip-debug"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "!clean_choice!"=="7" (
    echo Cleaning all build directories...
    if exist "build" rmdir /s /q "build"
    echo Done!
) else if "!clean_choice!"=="8" (
    goto MAIN_MENU
) else (
    echo Invalid selection!
)
pause
goto MAIN_MENU

:SETUP_VCPKG
cls
echo =====================================
echo   Setup vcpkg
echo =====================================
echo.

if exist "vcpkg\vcpkg.exe" (
    echo vcpkg is already installed!
    echo.
    set /p update_choice="Do you want to update vcpkg? (y/n): "
    if /i "!update_choice!"=="y" (
        cd vcpkg
        git pull
        .\bootstrap-vcpkg.bat
        cd ..
    )
) else (
    echo vcpkg not found. Installing...
    echo.

    :: Clone vcpkg
    git clone https://github.com/Microsoft/vcpkg.git

    :: Bootstrap vcpkg
    cd vcpkg
    call bootstrap-vcpkg.bat
    cd ..

    echo.
    echo vcpkg installed successfully!
)

echo.
echo Installing required packages...
echo This may take a while...
echo.

cd vcpkg
.\vcpkg install boost-beast boost-asio boost-system boost-thread boost-program-options boost-date-time boost-regex boost-random openssl curl zlib nlohmann-json --triplet x64-windows
cd ..

echo.
echo Package installation complete!
pause
goto MAIN_MENU

:CHECK_NVIDIA_TOOLS
:: Check for CUDA
if not exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" (
    echo WARNING: CUDA 12.9 not found at expected location!
    echo CUDA builds may fail.
    echo.

    :: Try to find any CUDA installation
    for /d %%i in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
        if exist "%%i\bin\nvcc.exe" (
            echo Found CUDA installation at: %%i
            echo Consider updating CMakePresets.json to use this version.
            echo.
        )
    )
)
exit /b 0

:CHECK_AMD_TOOLS
:: Set default ROCm path if not set
if "%ROCM_PATH%"=="" (
    :: Check common Windows ROCm installation paths
    if exist "C:\Program Files\AMD\ROCm\6.2" (
        set ROCM_PATH=C:\Program Files\AMD\ROCm\6.2
    ) else if exist "C:\ROCm" (
        set ROCM_PATH=C:\ROCm
    ) else (
        echo WARNING: ROCm not found! Please set ROCM_PATH environment variable.
        echo AMD/HIP builds will fail without ROCm installed.
        echo.
        echo Download ROCm from: https://www.amd.com/en/developer/rocm-software.html
        echo.
        pause
        exit /b 1
    )
)

echo Found ROCm at: %ROCM_PATH%
echo.
exit /b 0

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

:: Try to find and setup Visual Studio 2022 environment
echo Setting up Visual Studio environment...

:: First try VS2022 Community
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    echo Visual Studio 2022 Community environment loaded
    exit /b 0
)

:: Try VS2022 Professional
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    echo Visual Studio 2022 Professional environment loaded
    exit /b 0
)

:: Try VS2022 Enterprise
if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    echo Visual Studio 2022 Enterprise environment loaded
    exit /b 0
)

:: Try VS2022 BuildTools
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    echo Visual Studio 2022 Build Tools environment loaded
    exit /b 0
)

:: Try using vswhere to find VS installation
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" (
        call "%%i\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
        echo Visual Studio environment loaded from: %%i
        exit /b 0
    )
)

echo ERROR: Could not find Visual Studio 2022 installation!
echo Please install Visual Studio 2022 with C++ development tools
echo Or run this script from a "Developer Command Prompt for VS 2022"
pause
exit /b 1

:: Check for required tools at startup
where cmake >nul 2>nul
if errorlevel 1 (
    echo ERROR: CMake not found in PATH!
    echo Please install CMake and add it to your PATH.
    pause
    exit /b 1
)

where ninja >nul 2>nul
if errorlevel 1 (
    echo WARNING: Ninja not found in PATH!
    echo Ninja builds will not work without it.
    echo.
    echo You can install Ninja by:
    echo 1. Download from https://github.com/ninja-build/ninja/releases
    echo 2. Or install via winget: winget install Ninja-build.Ninja
    echo 3. Or install via chocolatey: choco install ninja
    echo.
)

:: Display current backend
echo Current GPU Backend: %GPU_BACKEND%
echo.
pause

:: Start main menu
goto MAIN_MENU
