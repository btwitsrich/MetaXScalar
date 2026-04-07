# push.ps1 — Stage, commit and push all local changes to GitHub
# Usage: .\push.ps1
#        .\push.ps1 "your custom commit message"

param(
    [string]$Message = ""
)

Set-Location $PSScriptRoot

# Check for any changes
$status = git status --porcelain
if (-not $status) {
    Write-Host "✅ Nothing to push — working tree is clean." -ForegroundColor Green
    exit 0
}

# Build commit message
if (-not $Message) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    $Message = "update: local changes [$timestamp]"
}

Write-Host "📦 Staging all changes..." -ForegroundColor Cyan
git add .

Write-Host "💾 Committing: $Message" -ForegroundColor Cyan
git commit -m $Message

Write-Host "🚀 Pushing to GitHub..." -ForegroundColor Cyan
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Done! Changes are live at: https://github.com/btwitsrich/MetaXScalar" -ForegroundColor Green
} else {
    Write-Host "`n❌ Push failed. Check the error above." -ForegroundColor Red
}
