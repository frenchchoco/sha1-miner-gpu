@echo off
setlocal EnableDelayedExpansion

REM SHA-1 OP_NET Miner - Windows Dependencies Installer
REM Supports NVIDIA and AMD GPUs, locale-agnostic, auto-installs dependencies

REM Set UTF-8 code page for locale-agnostic output
chcp 65001 >nul

REM Set colors for output
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "ERROR=[ERROR]"
set "WARNING=[WARNING]"

REM Default installation directory
if "%~1"=="" (
    set "INSTALL_DIR=%CD%"
) else (
    set "INSTALL_DIR=%~1"
)

REM Define URLs and versions for dependencies
set "NINJA_VERSION=1.12.1"
set "NINJA_URL=https://github.com/ninja-build/ninja/releases/download/v%NINJA_VERSION%/ninja-win.zip"
set "CMAKE_VERSION=3.28.3"
set "CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v%CMAKE_VERSION%/cmake-%CMAKE_VERSION%-windows-x86_64.msi"
set "PYTHON_VERSION=3.10.6"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe"
set "TEMP_DIR=%TEMP%\SHA1_Miner_Install"

REM Check for administrative privileges
net session >nul 2>&1
if errorlevel 1 (
    echo.
    echo ========================================
    echo %ERROR% Administrator rights required!
    echo ========================================
    echo Please run this script as Administrator:
    echo 1. Right-click on install.bat
    echo 2. Select "Run as administrator"
    echo.
    pause
    goto :EOF
)

REM Header
cls
echo =====================================
echo SHA-1 Miner - Dependencies Installer
echo =====================================
echo Running as Administrator: YES
echo Working directory: %INSTALL_DIR%
echo.

REM Create temporary directory
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

REM Install Ninja to INSTALL_DIR\ninja
echo %INFO% Checking for Ninja...
where ninja >nul 2>&1
if errorlevel 1 (
    echo %INFO% Ninja not found. Downloading Ninja %NINJA_VERSION%...
    powershell -Command "Invoke-WebRequest -Uri '%NINJA_URL%' -OutFile '%TEMP_DIR%\ninja-win.zip'" >nul 2>&1
    if exist "%TEMP_DIR%\ninja-win.zip" (
        echo %INFO% Extracting Ninja to %INSTALL_DIR%\ninja...
        if not exist "%INSTALL_DIR%\ninja" mkdir "%INSTALL_DIR%\ninja"
        powershell -Command "Expand-Archive -Path '%TEMP_DIR%\ninja-win.zip' -DestinationPath '%INSTALL_DIR%\ninja' -Force" >nul 2>&1
        if exist "%INSTALL_DIR%\ninja\ninja.exe" (
            echo %SUCCESS% Ninja installed to %INSTALL_DIR%\ninja
            REM Add Ninja to system PATH using a more robust method
            echo %INFO% Adding %INSTALL_DIR%\ninja to system PATH...
            powershell -Command "$currentPath = [Environment]::GetEnvironmentVariable('PATH', 'Machine'); $newPath = [System.String]::Join(';', $currentPath.Split(';').Where({ $_ -ne '%INSTALL_DIR%\ninja' })); if (-not ($newPath -like '*%INSTALL_DIR%\ninja*')) { [Environment]::SetEnvironmentVariable('PATH', $newPath + ';%INSTALL_DIR%\ninja', 'Machine') }" >nul 2>&1
            set "PATH=%PATH%;%INSTALL_DIR%\ninja"
        ) else (
            echo %ERROR% Failed to extract Ninja.
            pause
            goto :EOF
        )
    ) else (
        echo %ERROR% Failed to download Ninja.
        echo Please download manually: https://github.com/ninja-build/ninja/releases
        pause
        goto :EOF
    )
) else (
    echo %SUCCESS% Ninja found.
    REM Ensure Ninja is in INSTALL_DIR\ninja
    for /f "tokens=*" %%i in ('where ninja') do (
        set "NINJA_PATH=%%~dpi"
        set "NINJA_PATH=!NINJA_PATH:~0,-1!"
        if not "!NINJA_PATH!"=="%INSTALL_DIR%\ninja" (
            echo %INFO% Moving existing Ninja to %INSTALL_DIR%\ninja...
            if not exist "%INSTALL_DIR%\ninja" mkdir "%INSTALL_DIR%\ninja"
            copy "%%i" "%INSTALL_DIR%\ninja\ninja.exe" >nul
            if exist "%INSTALL_DIR%\ninja\ninja.exe" (
                echo %SUCCESS% Ninja moved to %INSTALL_DIR%\ninja
                powershell -Command "$currentPath = [Environment]::GetEnvironmentVariable('PATH', 'Machine'); $newPath = [System.String]::Join(';', $currentPath.Split(';').Where({ $_ -ne '%INSTALL_DIR%\ninja' })); if (-not ($newPath -like '*%INSTALL_DIR%\ninja*')) { [Environment]::SetEnvironmentVariable('PATH', $newPath + ';%INSTALL_DIR%\ninja', 'Machine') }" >nul 2>&1
                set "PATH=%PATH%;%INSTALL_DIR%\ninja"
            ) else (
                echo %WARNING% Failed to move Ninja. Continuing...
            )
        )
    )
)
echo.

REM Check and add existing Git to PATH
echo %INFO% Checking for Git...
call :CHECK_GIT
if errorlevel 1 (
    echo %ERROR% Git not found! Please install Git manually:
    echo            https://git-scm.com/download/win
    pause
    goto :EOF
)
echo.

REM Check and install Python
echo %INFO% Checking for Python...
where python3 >nul 2>&1
if errorlevel 1 (
    echo %INFO% Python not found. Downloading Python %PYTHON_VERSION%...
    powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%TEMP_DIR%\python-installer.exe'" >nul 2>&1
    if exist "%TEMP_DIR%\python-installer.exe" (
        echo %INFO% Installing Python...
        start /wait "" "%TEMP_DIR%\python-installer.exe" /quiet InstallAllUsers=1 PrependPath=1
        if errorlevel 0 (
            echo %SUCCESS% Python installed.
        ) else (
            echo %ERROR% Failed to install Python.
            pause
            goto :EOF
        )
    ) else (
        echo %ERROR% Failed to download Python installer.
        pause
        goto :EOF
    )
) else (
    echo %SUCCESS% Python found.
)
echo.

REM Check and install CMake
echo %INFO% Checking for CMake...
where cmake >nul 2>&1
if errorlevel 1 (
    echo %INFO% CMake not found. Downloading CMake %CMAKE_VERSION%...
    powershell -Command "Invoke-WebRequest -Uri '%CMAKE_URL%' -OutFile '%TEMP_DIR%\cmake-installer.msi'" >nul 2>&1
    if exist "%TEMP_DIR%\cmake-installer.msi" (
        echo %INFO% Installing CMake...
        msiexec /i "%TEMP_DIR%\cmake-installer.msi" /quiet /norestart ADD_CMAKE_TO_PATH=System
        if errorlevel 0 (
            echo %SUCCESS% CMake installed.
        ) else (
            echo %ERROR% Failed to install CMake.
            pause
            goto :EOF
        )
    ) else (
        echo %ERROR% Failed to download CMake installer.
        pause
        goto :EOF
    )
) else (
    echo %SUCCESS% CMake found.
)
REM Add manual CMake path update as fallback
where cmake >nul 2>&1
if errorlevel 1 (
    for /f "tokens=2*" %%a in ('reg query "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{%CMAKE_VERSION%-windows-x86_64}" /v InstallLocation 2^>nul') do (
        set "CMAKE_PATH=%%b\bin"
        if exist "!CMAKE_PATH!\cmake.exe" (
            echo %INFO% Adding !CMAKE_PATH! to system PATH as fallback...
            powershell -Command "$currentPath = [Environment]::GetEnvironmentVariable('PATH', 'Machine'); $newPath = [System.String]::Join(';', $currentPath.Split(';').Where({ $_ -ne '!CMAKE_PATH!' })); if (-not ($newPath -like '*!CMAKE_PATH!*')) { [Environment]::SetEnvironmentVariable('PATH', $newPath + ';!CMAKE_PATH!', 'Machine') }" >nul 2>&1
            set "PATH=%PATH%;!CMAKE_PATH!"
            echo %SUCCESS% CMake path updated.
        )
    )
)
echo.

REM Check for Visual Studio
echo %INFO% Checking for Visual Studio...
set "VS_FOUND=0"
for %%v in (Community Professional Enterprise BuildTools) do (
    if exist "%ProgramFiles%\Microsoft Visual Studio\2022\%%v\VC\Auxiliary\Build\vcvars64.bat" (
        set "VS_FOUND=1"
        set "VS_PATH=%ProgramFiles%\Microsoft Visual Studio\2022\%%v"
    )
)
if "%VS_FOUND%"=="0" (
    echo %WARNING% Visual Studio 2022 not found. Please install:
    echo            https://visualstudio.microsoft.com/downloads/
    echo            (Select 'Desktop development with C++' workload)
) else (
    echo %SUCCESS% Visual Studio 2022 found at: %VS_PATH%
)
echo.

REM Detect GPU type (locale-agnostic) - IMPROVED
set "GPU_TYPE=NONE"
set "HAS_NVIDIA=0"
set "HAS_AMD=0"
echo %INFO% Detecting installed GPUs...
powershell -Command "$gpus = Get-WmiObject -Class Win32_VideoController; foreach ($gpu in $gpus) { Write-Host \"[INFO] Found Device ID: \" $gpu.PNPDeviceID; if ($gpu.PNPDeviceID -match 'VEN_10DE') { Write-Host \"[SUCCESS] NVIDIA GPU detected.\"; [System.Environment]::SetEnvironmentVariable('GPU_TYPE', 'NVIDIA', 'Process'); } if ($gpu.PNPDeviceID -match 'VEN_1002') { Write-Host \"[SUCCESS] AMD Radeon GPU detected.\"; [System.Environment]::SetEnvironmentVariable('GPU_TYPE', 'AMD', 'Process'); } }"
if not "%GPU_TYPE%"=="NONE" (
    echo %SUCCESS% GPU type set to: %GPU_TYPE%
    if "%GPU_TYPE%"=="NVIDIA" (
        set "HAS_NVIDIA=1"
    ) else if "%GPU_TYPE%"=="AMD" (
        set "HAS_AMD=1"
    )
) else (
    echo %WARNING% No supported GPU detected.
    echo            The miner requires either NVIDIA or AMD GPU.
    echo            Continuing with dependency installation...
)
echo.

REM Determine vcpkg triplet
set "VCPKG_TRIPLET=x64-windows"
if "%GPU_TYPE%"=="AMD" (
    echo %INFO% AMD GPU detected - using static libraries
    set "VCPKG_TRIPLET=x64-windows-static"
)

REM GPU-specific checks
if "%HAS_NVIDIA%"=="1" (
    echo %INFO% Checking for CUDA...
    where nvcc >nul 2>&1
    if errorlevel 1 (
        echo %WARNING% CUDA not found. Please install CUDA Toolkit:
        echo            https://developer.nvidia.com/cuda-downloads
        echo            (Version 12.9 recommended)
    ) else (
        for /f "tokens=*" %%i in ('nvcc --version') do (
            echo %SUCCESS% CUDA found: %%i
        )
    )
)
if "%HAS_AMD%"=="1" (
    echo %INFO% Checking for AMD ROCm...
    set "ROCM_FOUND=0"
    if exist "%ProgramFiles%\AMD\ROCm" (
        for /d %%v in ("%ProgramFiles%\AMD\ROCm\*") do (
            if exist "%%v\bin\hipcc.exe" (
                set "ROCM_PATH=%%v"
                set "ROCM_FOUND=1"
            )
        )
    )
    if "%ROCM_FOUND%"=="1" (
        echo %SUCCESS% AMD ROCm found at: !ROCM_PATH!
        setx ROCM_PATH "!ROCM_PATH!" >nul
    ) else (
        echo %WARNING% AMD ROCm not found. Please install:
        echo            https://rocm.docs.amd.com/en/latest/deploy/windows/index.html
        echo            (Version 6.2 or later recommended)
    )
)
echo.

REM Setup vcpkg
cd /d "%INSTALL_DIR%"
if exist vcpkg.json (
    echo %INFO% Removing existing vcpkg.json...
    del vcpkg.json
)
echo %INFO% Setting up vcpkg...
if exist "vcpkg\vcpkg.exe" (
    echo %INFO% vcpkg found, updating...
    cd vcpkg
    git pull >nul
    call bootstrap-vcpkg.bat
    cd ..
) else (
    echo %INFO% Cloning vcpkg...
    git clone https://github.com/Microsoft/vcpkg.git
    if errorlevel 1 (
        echo %ERROR% Failed to clone vcpkg.
        pause
        goto :EOF
    )
    cd vcpkg
    call bootstrap-vcpkg.bat
    cd ..
)
echo.

REM Install vcpkg dependencies
echo %INFO% Installing C++ dependencies (triplet: %VCPKG_TRIPLET%)...
echo %INFO% This may take 10-30 minutes...
set "DEPENDENCIES=openssl boost-beast boost-asio boost-system boost-thread boost-program-options boost-date-time boost-regex boost-random boost-chrono boost-atomic nlohmann-json zlib"
for %%d in (%DEPENDENCIES%) do (
    echo %INFO% Installing %%d...
    vcpkg\vcpkg install %%d:%VCPKG_TRIPLET%
    if errorlevel 1 (
        echo %WARNING% %%d installation had issues, continuing...
    )
)
echo.

REM Integrate vcpkg
echo %INFO% Integrating vcpkg...
vcpkg\vcpkg integrate install

REM Set environment variables
echo %INFO% Setting OPENSSL_ROOT_DIR...
set "OPENSSL_ROOT_DIR=%INSTALL_DIR%\vcpkg\installed\%VCPKG_TRIPLET%"
setx OPENSSL_ROOT_DIR "%OPENSSL_ROOT_DIR%" >nul

REM Verify installations
echo %INFO% Verifying installations...
where ninja >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Ninja not found.
) else (
    echo %SUCCESS% Ninja found.
    for /f "tokens=*" %%i in ('ninja --version') do (
        echo %INFO% Ninja version: %%i
    )
)
where cmake >nul 2>&1
if errorlevel 1 (
    echo %ERROR% CMake not found.
) else (
    echo %SUCCESS% CMake found.
    for /f "tokens=*" %%i in ('cmake --version') do (
        echo %INFO% CMake version: %%i
        goto :CMAKE_VERSION_DONE
    )
)
:CMAKE_VERSION_DONE
where git >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Git not found.
) else (
    echo %SUCCESS% Git found.
    for /f "tokens=*" %%i in ('git --version') do (
        echo %INFO% Git version: %%i
    )
)
where python3 >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Python not found.
) else (
    echo %SUCCESS% Python found.
    for /f "tokens=*" %%i in ('python3 --version') do (
        echo %INFO% Python version: %%i
    )
)
echo %INFO% Checking Boost version...
vcpkg\vcpkg list boost | findstr /i "boost"
echo.

REM Cleanup temporary files
echo %INFO% Cleaning up temporary files...
rd /s /q "%TEMP_DIR%" 2>nul

REM Summary
echo.
echo =====================================
echo Dependencies Installation Complete!
echo =====================================
echo.
echo Installed packages (triplet: %VCPKG_TRIPLET%):
echo   - Ninja
echo   - CMake
echo   - Git
echo   - Python
echo   - OpenSSL
echo   - Boost 1.88 (with Beast, system, thread, etc.)
echo   - nlohmann-json
echo   - zlib
echo.
echo GPU Support Status:
if "%HAS_NVIDIA%"=="1" (
    echo   - NVIDIA GPU: Detected
    where nvcc >nul 2>&1
    if errorlevel 1 (
        echo   - CUDA: Not installed
    ) else (
        echo   - CUDA: Installed
        for /f "tokens=*" %%i in ('nvcc --version') do (
            echo   - CUDA version: %%i
        )
    )
)
if "%HAS_AMD%"=="1" (
    echo   - AMD GPU: Detected
    if "%ROCM_FOUND%"=="1" (
        echo   - ROCm: Installed at !ROCM_PATH!
    ) else (
        echo   - ROCm: Not installed
    )
)
if "%GPU_TYPE%"=="NONE" (
    echo   - No supported GPU detected
)
echo.
echo Next steps:
echo   1. Ensure all required tools are installed:
if "%VS_FOUND%"=="0" (
    echo      - Install Visual Studio 2022 (Desktop development with C++)
)
if "%HAS_NVIDIA%"=="1" (
    where nvcc >nul 2>&1
    if errorlevel 1 (
        echo      - Install CUDA Toolkit
    )
)
if "%HAS_AMD%"=="1" if "%ROCM_FOUND%"=="0" (
    echo      - Install AMD ROCm (version 6.2 or later)
)
echo   2. Configure the project:
if "%HAS_NVIDIA%"=="1" (
    echo      build.bat
)
if "%HAS_AMD%"=="1" (
    echo      build.bat
)
echo   3. Build the project:
if "%HAS_NVIDIA%"=="1" (
    echo      build.bat
)
if "%HAS_AMD%"=="1" (
    echo      build.bat
)
echo.
echo Press any key to exit...
pause >nul
goto :EOF

:CHECK_GIT
where git >nul 2>&1
if not errorlevel 1 (
    echo %SUCCESS% Git found.
    for /f "tokens=*" %%i in ('git --version') do (
        echo %INFO% Git version: %%i
    )
    exit /b 0
)

echo %INFO% Git not found in PATH. Searching for existing installation...

REM Check common Git installation paths using a safer if/else chain
if exist "C:\Program Files\Git\bin\git.exe" (
    set "GIT_PATH=C:\Program Files\Git\bin"
    goto :GIT_FOUND
)
if exist "C:\Program Files (x86)\Git\bin\git.exe" (
    set "GIT_PATH=C:\Program Files (x86)\Git\bin"
    goto :GIT_FOUND
)
if exist "%USERPROFILE%\AppData\Local\Programs\Git\bin\git.exe" (
    set "GIT_PATH=%USERPROFILE%\AppData\Local\Programs\Git\bin"
    goto :GIT_FOUND
)

REM Check registry for Git installation
for /f "tokens=2*" %%a in ('reg query "HKLM\SOFTWARE\GitForWindows" /v InstallPath 2^>nul') do (
    if exist "%%b\bin\git.exe" (
        set "GIT_PATH=%%b\bin"
        goto :GIT_FOUND
    )
)

echo %ERROR% Git not found!
exit /b 1

:GIT_FOUND
echo %SUCCESS% Git found at: !GIT_PATH!
echo %INFO% Adding !GIT_PATH! to system PATH...
powershell -Command "$currentPath = [Environment]::GetEnvironmentVariable('PATH', 'Machine'); $newPath = [System.String]::Join(';', $currentPath.Split(';').Where({ $_ -ne '!GIT_PATH!' })); if (-not ($newPath -like '*!GIT_PATH!*')) { [Environment]::SetEnvironmentVariable('PATH', $newPath + ';!GIT_PATH!', 'Machine') }" >nul 2>&1
set "PATH=%PATH%;!GIT_PATH!"
exit /b 0
