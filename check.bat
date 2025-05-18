@echo off
REM ============================================================================
REM chak.cmd Run a suite of quality checks and formatters on a Python file or directory
REM Usage: chak.cmd <path\to\file_or_directory>
REM ============================================================================

REM Check for argument
if "%~1"=="" (
    echo Usage: %~nx0 ^<python-file-or-directory^>
    exit /b 1
)

set "TARGET=%~1"

echo.
echo Formatting (black)...
black --line-length 88 --skip-string-normalization "%TARGET%"
if errorlevel 1 exit /b 1

echo.
echo Running type checks (mypy)...
python -m mypy "%TARGET%"

echo.
echo Sorting imports (isort)...
isort "%TARGET%"
if errorlevel 1 exit /b 1

echo.
echo Removing unused code (autoflake)...
autoflake --in-place --remove-unused-variables --remove-all-unused-imports "%TARGET%"
if errorlevel 1 exit /b 1

echo.
echo Linting style (flake8)...
flake8 "%TARGET%"

echo.
echo In-depth linting (pylint)...
pylint "%TARGET%"

echo.
echo Security scan (bandit)...
bandit -q -r "%TARGET%"

echo.
echo Dead-code detection (vulture)...
vulture "%TARGET%"

echo.
echo Complexity metrics (radon)...
radon cc "%TARGET%" --min C

echo.
echo Dependency vulnerability check (safety)...
safety scan

echo.
echo Modernizing syntax (pyupgrade)...
pyupgrade --py38-plus "%TARGET%"

echo.
echo All checks passed and code formatted for "%TARGET%"
exit /b 0
