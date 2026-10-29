@echo off
echo ═══════════════════════════════════════════════════════════
echo    DEEPGUARD - Demo Files Setup
echo ═══════════════════════════════════════════════════════════
echo.
echo This will copy demo files to your Desktop for easy access
echo during your presentation.
echo.
pause

REM Create demo folder on Desktop
set DESKTOP=%USERPROFILE%\Desktop
set DEMO_FOLDER=%DESKTOP%\DeepGuard_Demo_Files

echo.
echo Creating demo folder...
mkdir "%DEMO_FOLDER%" 2>nul

echo Copying sample files...

REM Copy sample files
copy "sample_data\sample_real_face.jpg" "%DEMO_FOLDER%\1_REAL_Face.jpg"
copy "sample_data\sample_deepfake_face.jpg" "%DEMO_FOLDER%\2_FAKE_Face.jpg"
copy "sample_data\sample_video.mp4" "%DEMO_FOLDER%\3_Video.mp4"
copy "sample_data\sample_real_speech.wav" "%DEMO_FOLDER%\4_REAL_Audio.wav"
copy "sample_data\sample_voice_clone.wav" "%DEMO_FOLDER%\5_FAKE_Audio.wav"

REM Copy documentation
copy "VIVA_PREPARATION_GUIDE.md" "%DEMO_FOLDER%\"
copy "PROJECT_DESCRIPTION.md" "%DEMO_FOLDER%\"
copy "PRESENTATION_CHEAT_SHEET.txt" "%DEMO_FOLDER%\"

echo.
echo ═══════════════════════════════════════════════════════════
echo ✓ Demo files copied to: %DEMO_FOLDER%
echo.
echo Files ready for presentation:
echo   • 1_REAL_Face.jpg (test real image)
echo   • 2_FAKE_Face.jpg (test deepfake image)
echo   • 3_Video.mp4 (test video)
echo   • 4_REAL_Audio.wav (test real audio)
echo   • 5_FAKE_Audio.wav (test fake audio)
echo   • All documentation files
echo.
echo ═══════════════════════════════════════════════════════════
echo.
pause

REM Open the folder
start "" "%DEMO_FOLDER%"
