@echo off
echo ═══════════════════════════════════════════════════════════
echo    DeepGuard - GitHub Upload Script
echo ═══════════════════════════════════════════════════════════
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git is not installed!
    echo.
    echo Please install Git from: https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

echo ✓ Git is installed
echo.

REM Check if git config is set
git config user.name >nul 2>&1
if errorlevel 1 (
    echo.
    echo ⚠️  Git is not configured yet!
    echo.
    echo Please run these commands first:
    echo   git config --global user.name "Your Name"
    echo   git config --global user.email "your.email@example.com"
    echo.
    pause
    exit /b 1
)

echo ✓ Git is configured
echo   User: 
git config user.name
echo   Email: 
git config user.email
echo.

echo ══════════════════════════════════════════════════════════
echo  STEP 1: Initialize Git Repository
echo ══════════════════════════════════════════════════════════
echo.

REM Check if already initialized
if exist ".git" (
    echo ℹ️  Git already initialized, skipping...
) else (
    git init
    echo ✓ Git repository initialized
)

echo.
echo ══════════════════════════════════════════════════════════
echo  STEP 2: Stage All Files
echo ══════════════════════════════════════════════════════════
echo.

git add .
echo ✓ All files staged

echo.
echo Files to be committed:
git status --short
echo.

pause

echo.
echo ══════════════════════════════════════════════════════════
echo  STEP 3: Create Commit
echo ══════════════════════════════════════════════════════════
echo.

git commit -m "Initial commit: DeepGuard deepfake detection system with 99.94%% accuracy"
echo ✓ Commit created

echo.
echo ══════════════════════════════════════════════════════════
echo  STEP 4: Add Remote Repository
echo ══════════════════════════════════════════════════════════
echo.
echo Please enter your GitHub repository URL:
echo Example: https://github.com/YOUR_USERNAME/deepguard-detection.git
echo.
set /p REPO_URL="Repository URL: "

if "%REPO_URL%"=="" (
    echo ❌ No URL provided!
    pause
    exit /b 1
)

REM Remove origin if it exists
git remote remove origin >nul 2>&1

git remote add origin %REPO_URL%
echo ✓ Remote repository added

echo.
echo ══════════════════════════════════════════════════════════
echo  STEP 5: Push to GitHub
echo ══════════════════════════════════════════════════════════
echo.

git branch -M main
echo Pushing to GitHub...
echo.

git push -u origin main

if errorlevel 1 (
    echo.
    echo ❌ Push failed!
    echo.
    echo Common issues:
    echo  1. Wrong repository URL
    echo  2. Authentication required (need personal access token)
    echo  3. Repository doesn't exist yet
    echo.
    echo Please check the GITHUB_UPLOAD_GUIDE.txt for help
    pause
    exit /b 1
)

echo.
echo ═══════════════════════════════════════════════════════════
echo  ✅ SUCCESS! Project uploaded to GitHub!
echo ═══════════════════════════════════════════════════════════
echo.
echo Your repository: %REPO_URL%
echo.
echo Next steps:
echo  1. Visit your GitHub repository
echo  2. Add topics/tags for discoverability
echo  3. Add a LICENSE file
echo  4. Share your project!
echo.
echo ═══════════════════════════════════════════════════════════
pause
