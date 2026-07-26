# Clean evaluation script
Write-Host "Starting fresh evaluation..." -ForegroundColor Green
Write-Host "This will evaluate 6 models with 3 pipelines (direct, NLLB, M2M100)" -ForegroundColor Cyan
Write-Host "Estimated time: 20-30 minutes" -ForegroundColor Yellow
Write-Host ""

python evaluate.py --test
