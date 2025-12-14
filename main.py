#!/usr/bin/env python3
"""
DeepSeek Desktop - Aplicación principal
Versión con caché local para funcionar sin internet
"""

import sys
import os
import json
import time
from pathlib import Path

# Añadir directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui_app import DeepSeekApp

def check_dependencies():
    """Verifica dependencias instaladas"""
    required_packages = [
        'torch',
        'transformers',
        'customtkinter',
        'PIL',
        'sentencepiece'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    return missing

def create_default_config():
    """Crea archivo de configuración por defecto"""
    config = {
        "model": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "device": "cpu",
        "quantization": "none",
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "theme": "dark",
        "language": "es",
        "auto_save": True,
        "model_cache_dir": "./models",
        "local_model_path": "",
        "use_local_cache": True,
        "offline_mode": True
    }
    
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return config

def ensure_directories():
    """Crea directorios necesarios"""
    directories = ['models', 'data', 'data/cache', 'data/conversations', 'local_models']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def check_model_exists(config):
    """Verifica si el modelo ya existe localmente"""
    model_path = Path("models")
    
    # Buscar archivos de modelo
    safetensors_files = list(model_path.glob("**/*.safetensors"))
    bin_files = list(model_path.glob("**/*.bin"))
    
    # Verificar archivos esenciales
    essential_files = [
        "config.json",
        "tokenizer_config.json",
        "generation_config.json"
    ]
    
    all_essential_exist = True
    for file in essential_files:
        if not list(model_path.glob(f"**/{file}")):
            all_essential_exist = False
            break
    
    return len(safetensors_files) > 0 or len(bin_files) > 0, all_essential_exist

def main():
    """Función principal"""
    print("=" * 50)
    print("🤖 DeepSeek Desktop - IA Local")
    print("Materia: Inteligencia Artificial")
    print("=" * 50)
    
    # Verificar dependencias
    missing = check_dependencies()
    if missing:
        print(f"\n❌ Faltan dependencias: {', '.join(missing)}")
        print("Ejecuta: pip install " + " ".join(missing))
        sys.exit(1)
    
    # Crear directorios
    ensure_directories()
    
    # Configuración
    if not os.path.exists('config.json'):
        print("\n📝 Creando configuración por defecto...")
        config = create_default_config()
    else:
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
        except:
            config = create_default_config()
    
    # Verificar si modelo existe localmente
    has_model, has_all_files = check_model_exists(config)
    
    print("\n✅ Sistema preparado:")
    print(f"   Modelo: {config['model']}")
    print(f"   Dispositivo: {config['device']}")
    print(f"   Tema: {config['theme']}")
    
    if has_model:
        if has_all_files:
            print("   📦 Modelo disponible localmente")
        else:
            print("   ⚠️  Modelo parcialmente descargado")
            print("   Conecta a internet para completar descarga")
    else:
        print("   ❌ Modelo no encontrado localmente")
        print("   Conecta a internet para descargar por primera vez")
    
    # Iniciar aplicación
    print("\n🚀 Iniciando interfaz gráfica...")
    app = DeepSeekApp(config)
    app.run()

if __name__ == "__main__":
    main()