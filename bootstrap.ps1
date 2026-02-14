Write-Host "SEAI dependency setup starting..."

# Must run inside repo root
if (!(Test-Path requirements.txt)) {
    Write-Host "requirements.txt not found. Run this inside the project root."
    exit
}

# Create virtual environment if missing
if (!(Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
} else {
    Write-Host "Virtual environment already exists."
}

Write-Host "Activating virtual environment..."
.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt ..."
pip install -r requirements.txt

Write-Host "Running import test..."
python -c "import torch, numpy, pandas, river, sklearn; print('Import test passed')"

Write-Host "Setup complete."
Write-Host "To activate later run: .venv\Scripts\Activate.ps1"
