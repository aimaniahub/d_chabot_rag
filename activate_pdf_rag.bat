@echo off
cd /d "%~dp0"
call C:\Users\prasa\anaconda3\Scripts\activate.bat pdf_rag
echo Ready: pdf_rag
echo   python app.py ingest ^| chat ^| serve ^| eval ^| ui
