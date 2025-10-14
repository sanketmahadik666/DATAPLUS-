@echo off
echo Converting HTML documents to PDF using browser automation...
echo.

REM Check if Chrome is available
where chrome >nul 2>nul
if %errorlevel% equ 0 (
    echo Using Google Chrome for PDF conversion...
    goto :chrome_convert
)

REM Check if Edge is available
where msedge >nul 2>nul
if %errorlevel% equ 0 (
    echo Using Microsoft Edge for PDF conversion...
    goto :edge_convert
)

echo No suitable browser found for automated conversion.
echo Please manually open the HTML files and use Print to PDF.
echo.
echo Opening the index file for manual conversion...
start "" "procurement_documents_html\index.html"
goto :end

:chrome_convert
echo Converting documents using Chrome...
for /r "procurement_documents_html" %%f in (*.html) do (
    if not "%%~nxf"=="index.html" (
        echo Converting %%~nxf...
        chrome --headless --disable-gpu --print-to-pdf="%%~dpnf.pdf" "%%f"
    )
)
goto :end

:edge_convert
echo Converting documents using Edge...
for /r "procurement_documents_html" %%f in (*.html) do (
    if not "%%~nxf"=="index.html" (
        echo Converting %%~nxf...
        msedge --headless --disable-gpu --print-to-pdf="%%~dpnf.pdf" "%%f"
    )
)

:end
echo.
echo PDF conversion complete!
echo Check the procurement_documents_html directory for PDF files.
pause
