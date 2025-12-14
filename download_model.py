#!/usr/bin/env python3
"""
Script para descargar modelo una vez y guardarlo localmente
Ejecutar CON INTERNET antes de usar offline
"""

import sys
import os
import json
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model(model_name="deepseek-ai/deepseek-coder-1.3b-instruct", 
                   cache_dir="./models"):
    """
    Descarga el modelo y lo guarda localmente
    
    Args:
        model_name: Nombre del modelo en HuggingFace
        cache_dir: Directorio donde guardar
    """
    
    print("=" * 60)
    print("📥 DESCARGADOR DE MODELO DeepSeek")
    print("=" * 60)
    print(f"Modelo: {model_name}")
    print(f"Guardar en: {cache_dir}")
    print("=" * 60)
    
    # Crear directorio
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Configurar para descarga completa
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '0'
    
    try:
        # Paso 1: Descargar tokenizador
        print("\n1️⃣ Descargando tokenizador...")
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_path),
            local_files_only=False,
            trust_remote_code=True
        )
        
        # Forzar guardado completo
        tokenizer.save_pretrained(cache_path)
        
        tokenizer_time = time.time() - start_time
        print(f"   ✅ Tokenizador descargado ({tokenizer_time:.1f}s)")
        
        # Paso 2: Descargar modelo
        print("\n2️⃣ Descargando modelo (esto puede tomar varios minutos)...")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_path),
            local_files_only=False,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        
        # Forzar guardado completo
        model.save_pretrained(cache_path, safe_serialization=True)
        
        model_time = time.time() - start_time
        print(f"   ✅ Modelo descargado ({model_time:.1f}s)")
        
        # Paso 3: Verificar archivos descargados
        print("\n3️⃣ Verificando archivos descargados...")
        files = list(cache_path.glob("**/*"))
        print(f"   📁 {len(files)} archivos en total")
        
        # Mostrar archivos principales
        essential_files = ["config.json", "tokenizer_config.json", 
                          "model.safetensors", "pytorch_model.bin"]
        
        for file in essential_files:
            if list(cache_path.glob(f"**/{file}")):
                print(f"   ✅ {file}")
            else:
                print(f"   ⚠️  {file} (no encontrado)")
        
        # Crear archivo de verificación
        verification = {
            "model": model_name,
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cache_path": str(cache_path.absolute()),
            "tokenizer_time": tokenizer_time,
            "model_time": model_time,
            "total_time": tokenizer_time + model_time,
            "status": "complete"
        }
        
        with open(cache_path / "download_info.json", 'w', encoding='utf-8') as f:
            json.dump(verification, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print("🎉 ¡DESCARGA COMPLETADA!")
        print("=" * 60)
        print(f"Total tiempo: {verification['total_time']:.1f} segundos")
        print(f"Modelo guardado en: {cache_path.absolute()}")
        print("\nAhora puedes usar el programa SIN INTERNET.")
        print("Ejecuta: python main.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR en la descarga: {str(e)}")
        print("\nPosibles soluciones:")
        print("1. Verifica tu conexión a internet")
        print("2. Asegúrate de tener suficiente espacio en disco (al menos 5GB)")
        print("3. Intenta con un modelo más pequeño")
        print("4. Revisa que Python y pip estén actualizados")
        
        # Guardar error
        error_info = {
            "model": model_name,
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(cache_path / "download_error.json", 'w', encoding='utf-8') as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)
        
        return False

def show_menu():
    """Muestra menú de modelos disponibles"""
    print("\n📋 MODELOS DISPONIBLES:")
    print("1. deepseek-ai/deepseek-coder-1.3b-instruct (Recomendado - 2.7GB)")
    print("2. microsoft/phi-2 (Alternativa pequeña - 2.7GB)")
    print("3. google/gemma-2b (Alternativa buena - 2.5GB)")
    print("4. Otro (ingresa nombre completo)")
    print("0. Salir")
    
    choice = input("\nSelecciona opción (1-4): ").strip()
    
    models = {
        "1": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "2": "microsoft/phi-2",
        "3": "google/gemma-2b"
    }
    
    if choice == "0":
        print("Saliendo...")
        sys.exit(0)
    elif choice in models:
        return models[choice]
    elif choice == "4":
        custom_model = input("Ingresa el nombre del modelo en HuggingFace: ").strip()
        if custom_model:
            return custom_model
        else:
            print("Nombre inválido, usando opción por defecto")
            return "deepseek-ai/deepseek-coder-1.3b-instruct"
    else:
        print("Opción inválida, usando modelo por defecto")
        return "deepseek-ai/deepseek-coder-1.3b-instruct"

if __name__ == "__main__":
    print("🤖 DESCARGA INICIAL DE MODELO")
    print("Este script descarga el modelo UNA VEZ con internet")
    print("Luego podrás usar el programa sin conexión")
    print("\n" + "=" * 60)
    
    # Mostrar menú
    model_name = show_menu()
    
    # Confirmar
    print(f"\n¿Descargar {model_name}?")
    print("Necesitas aproximadamente 5GB de espacio libre.")
    confirm = input("Continuar? (s/n): ").strip().lower()
    
    if confirm != 's':
        print("Descarga cancelada")
        sys.exit(0)
    
    # Descargar
    success = download_model(model_name)
    
    if success:
        print("\n🎯 Instrucciones para usar sin internet:")
        print("1. Edita config.json y asegúrate que tenga:")
        print('   "model": "{}"'.format(model_name))
        print('   "offline_mode": true')
        print("2. Ejecuta: python main.py")
        print("3. ¡Disfruta de tu IA local!")
    else:
        print("\n❌ La descarga falló. Revisa los mensajes de error.")
    
    input("\nPresiona Enter para salir...")