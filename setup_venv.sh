#!/bin/bash
VENV_NAME="$(basename "$(pwd)")_venv"
python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt found"
fi

echo "Virtual environment '$VENV_NAME' created and activated"

pip install ipykernel
if jupyter kernelspec list | grep -q "$VENV_NAME"; then
    jupyter kernelspec uninstall -f "$VENV_NAME"
fi
python -m ipykernel install --user --name "$VENV_NAME" --display-name "Python ($VENV_NAME)"
echo "Jupyter kernel '$VENV_NAME' installed and ready to use"

exec "$SHELL"