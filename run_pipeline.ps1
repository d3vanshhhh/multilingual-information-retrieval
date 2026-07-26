# PowerShell version of run_pipeline.sh for Windows

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Multilingual IR - Semester VIII Pipeline" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 1: Generate embeddings" -ForegroundColor Yellow
Write-Host "----------------------------" -ForegroundColor Yellow
python generate_embeddings.py --test
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in embedding generation!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2: Mine hard negatives" -ForegroundColor Yellow
Write-Host "----------------------------" -ForegroundColor Yellow
python negative_mining.py --test
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in negative mining!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 3: Fine-tune model" -ForegroundColor Yellow
Write-Host "----------------------------" -ForegroundColor Yellow
python finetune.py --epochs 1 --test
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in fine-tuning!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 4: Run full evaluation" -ForegroundColor Yellow
Write-Host "----------------------------" -ForegroundColor Yellow
python evaluate.py --test
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in evaluation!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Pipeline complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To launch the Streamlit demo, run:" -ForegroundColor Cyan
Write-Host "  streamlit run streamlit_app.py" -ForegroundColor White
