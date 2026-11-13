# deploy_static_hf.ps1 — Push static web app to Hugging Face Static Space
# Usage: .\deploy_static_hf.ps1

$HF_USERNAME = "Jeeva-dev"
$SPACE_NAME = "CREDENCEX"
$STATIC_DIR = "d:\CREDENCEX\static_deploy"

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host "  CREDENCEX -> Hugging Face Static Space Deployment" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# Push using temporary git inside static_deploy
Push-Location $STATIC_DIR

try {
    Write-Host "`n[1/3] Initializing deployment repository..." -ForegroundColor Yellow
    if (-not (Test-Path ".git")) {
        git init -b main
    }
    git remote remove hf 2>$null
    git remote add hf "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

    Write-Host "[2/3] Adding files (index.html, styles.css, app.js, README.md)..." -ForegroundColor Yellow
    git add -A
    git commit -m "Deploy CREDENCEX interactive static dashboard" --allow-empty

    Write-Host "[3/3] Pushing to Hugging Face Space (main branch)..." -ForegroundColor Yellow
    Write-Host "Note: When prompted for password, enter your HF Access Token (write permission)." -ForegroundColor Gray
    git push hf main --force

    Write-Host "`n=======================================================" -ForegroundColor Green
    Write-Host " SUCCESS! Your live app is available at:" -ForegroundColor Green
    Write-Host " https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" -ForegroundColor Cyan
    Write-Host "=======================================================" -ForegroundColor Green
}
finally {
    Pop-Location
}
