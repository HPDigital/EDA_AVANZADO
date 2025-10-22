#!/bin/bash
# Script de inicio para CAF Dashboard con Tkinter (Linux/Mac)

echo "========================================"
echo "CAF Dashboard - Version Tkinter"
echo "========================================"
echo ""

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    echo "Activando entorno virtual..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activando entorno virtual..."
    source .venv/bin/activate
else
    echo "Advertencia: No se encontro entorno virtual"
fi

# Verificar instalacion de Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 no esta instalado"
    exit 1
fi

echo ""
echo "Iniciando CAF Dashboard..."
echo ""

# Ejecutar aplicacion
python3 app_tkinter.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error al ejecutar la aplicacion"
    echo "Verifique que las dependencias esten instaladas:"
    echo "  pip install -r requirements-tkinter.txt"
    echo ""
    read -p "Presione Enter para continuar..."
fi

