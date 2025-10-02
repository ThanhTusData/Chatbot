# Allow running script in current session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

Write-Host "🚀 Deploying Intelligent Chatbot..."

# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
. .\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download spaCy language models
python -m spacy download en_core_web_sm
python -m spacy download vi_core_news_sm

# 5. Create necessary directories
$dirs = @("models", "data", "logs", "templates", "static")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
}

# 6. Set environment variables
$env:FLASK_APP = "main.py"
$env:FLASK_ENV = "production"

# 7. Run database migrations (optional)
# python migrate.py

# 8. Start the chatbot application
Write-Host "✅ Starting chatbot server..."
python main.py

Write-Host "🎉 Deployment completed!"
Write-Host "🌐 Visit: http://localhost:5000"
