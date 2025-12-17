@echo off
REM SpecifyPlus Initialization Verification Script
REM Checks that all necessary components are properly set up

echo Verifying SpecifyPlus initialization...

REM Check if .specify directory exists
if not exist ".specify" (
    echo ERROR: .specify directory not found!
    exit /b 1
)

REM Check if memory directory exists and has constitution
if not exist ".specify\memory" (
    echo ERROR: .specify\memory directory not found!
    exit /b 1
)

if not exist ".specify\memory\constitution.md" (
    echo ERROR: .specify\memory\constitution.md not found!
    exit /b 1
)

REM Check if templates directory exists and has required templates
if not exist ".specify\templates" (
    echo ERROR: .specify\templates directory not found!
    exit /b 1
)

set template_count=0
for %%f in (".specify\templates\*.md") do set /a template_count+=1
if %template_count% LSS 5 (
    echo WARNING: Fewer than 5 templates found in .specify\templates
)

REM Check if scripts directory exists
if not exist ".specify\scripts" (
    echo WARNING: .specify\scripts directory not found
)

REM Check if commands directory exists
if not exist ".specify\commands" (
    echo WARNING: .specify\commands directory not found
)

REM Report success
echo.
echo SpecifyPlus initialization verification completed successfully!
echo.
echo Directory structure:
echo - .specify\memory (with constitution.md)
echo - .specify\templates (with %template_count% templates)
echo - .specify\scripts
echo - .specify\commands
echo.
echo Project is ready for Spec-Driven Development!