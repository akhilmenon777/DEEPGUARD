@echo off
echo ═══════════════════════════════════════════════════════════
echo    Creating Demo Images from Training Dataset
echo ═══════════════════════════════════════════════════════════
echo.

set DEMO_FOLDER=%USERPROFILE%\Desktop\DeepGuard_Demo_Images
set DATASET=data\140k_faces\real_vs_fake\real-vs-fake

echo Creating demo folder on Desktop...
mkdir "%DEMO_FOLDER%" 2>nul

echo.
echo Copying images from your training dataset...
echo.

REM Copy 3 real faces
if exist "%DATASET%\train\real\" (
    echo Copying REAL faces...
    xcopy "%DATASET%\train\real\*.jpg" "%DEMO_FOLDER%\REAL_*.jpg" /Y /I 2>nul
    
    REM Just copy first 3 files
    for /L %%i in (1,1,3) do (
        if exist "%DATASET%\train\real\real_0000%%i.jpg" (
            copy "%DATASET%\train\real\real_0000%%i.jpg" "%DEMO_FOLDER%\REAL_face_%%i.jpg"
        )
    )
)

REM Copy 3 fake faces
if exist "%DATASET%\train\fake\" (
    echo Copying FAKE faces...
    
    for /L %%i in (1,1,3) do (
        if exist "%DATASET%\train\fake\fake_0000%%i.jpg" (
            copy "%DATASET%\train\fake\fake_0000%%i.jpg" "%DEMO_FOLDER%\FAKE_face_%%i.jpg"
        )
    )
)

REM Also copy the sample images
echo.
echo Copying sample images...
copy "sample_data\sample_real_face.jpg" "%DEMO_FOLDER%\DEMO_REAL_sample.jpg" 2>nul
copy "sample_data\sample_deepfake_face.jpg" "%DEMO_FOLDER%\DEMO_FAKE_sample.jpg" 2>nul

echo.
echo ═══════════════════════════════════════════════════════════
echo ✓ Demo images ready!
echo.
echo Location: %DEMO_FOLDER%
echo.
echo Files created:
echo   • REAL_face_1.jpg, REAL_face_2.jpg, REAL_face_3.jpg
echo   • FAKE_face_1.jpg, FAKE_face_2.jpg, FAKE_face_3.jpg
echo   • DEMO_REAL_sample.jpg, DEMO_FAKE_sample.jpg
echo.
echo ═══════════════════════════════════════════════════════════
echo.
pause

start "" "%DEMO_FOLDER%"
