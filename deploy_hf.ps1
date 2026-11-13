# deploy_hf.ps1 — One-command deploy to HuggingFace Spaces
# Usage: .\deploy_hf.ps1

$HF_USERNAME = "Jeeva-dev"
$SPACE_NAME = "CREDENCEX"

Write-Host "`n=== CREDENCEX → HuggingFace Spaces Deployment ===" -ForegroundColor Cyan

# Step 1: Swap Dockerfile to the Spaces version
Write-Host "`n[1/4] Preparing Dockerfile for HuggingFace Spaces..." -ForegroundColor Yellow
Copy-Item -Path "Dockerfile" -Destination "Dockerfile.render.bak" -Force
Copy-Item -Path "Dockerfile.spaces" -Destination "Dockerfile" -Force

# Step 2: Swap README to HF Space format (keep original backed up)
Write-Host "[2/4] Setting up Space README metadata..." -ForegroundColor Yellow
Copy-Item -Path "README.md" -Destination "README_original.md" -Force
Copy-Item -Path "HF_README.md" -Destination "README.md" -Force

# Step 3: Add HuggingFace remote and push
Write-Host "[3/4] Adding HuggingFace remote..." -ForegroundColor Yellow
git remote remove hf 2>$null
git remote add hf "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

Write-Host "[4/4] Committing and pushing to HuggingFace Spaces..." -ForegroundColor Yellow
git add -A
git commit -m "deploy: CREDENCEX to HuggingFace Spaces"
git push hf main --force

Write-Host "`n✅ Deployed! Your app will be live at:" -ForegroundColor Green
Write-Host "   https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" -ForegroundColor Cyan

# Step 4: Restore original files for local/GitHub use
Write-Host "`n[Cleanup] Restoring original Dockerfile and README..." -ForegroundColor Yellow
Copy-Item -Path "Dockerfile.render.bak" -Destination "Dockerfile" -Force
Copy-Item -Path "README_original.md" -Destination "README.md" -Force
Remove-Item "Dockerfile.render.bak" -Force
Remove-Item "README_original.md" -Force
git add -A
git commit -m "chore: restore original Dockerfile and README after HF deploy" --allow-empty

Write-Host "`n🎉 Done! Local repo is back to normal." -ForegroundColor Green
