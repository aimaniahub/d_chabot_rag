# Activate conda env pdf_rag in this shell
& "C:\Users\prasa\anaconda3\shell\condabin\conda-hook.ps1"
conda activate pdf_rag
Set-Location $PSScriptRoot
Write-Host "Ready: pdf_rag | $(Get-Location)" -ForegroundColor Green
Write-Host "  python app.py ingest | chat | serve | eval | ui"
