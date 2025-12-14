#!/usr/bin/env python3
"""
Script para configurar modo offline
"""

import json
import sys

def set_offline_mode():
    """Configura el archivo config.json para modo offline"""
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}
    
    # Configurar para modo offline
    config["offline_mode"] = True
    config["use_local_cache"] = True
    config["model_cache_dir"] = "./models"
    config["fallback_to_gpt2"] = True
    
    # Guardar
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ Configuración actualizada para modo offline")
    print("📋 Configuración actual:")
    print(f"   Modo offline: {config.get('offline_mode', 'No configurado')}")
    print(f"   Usar caché local: {config.get('use_local_cache', 'No configurado')}")
    print(f"   Ruta caché: {config.get('model_cache_dir', 'No configurado')}")
    print(f"   Fallback a GPT-2: {config.get('fallback_to_gpt2', 'No configurado')}")

if __name__ == "__main__":
    print("🛠️ Configurando DeepSeek para modo offline")
    print("Este script configura la aplicación para funcionar sin internet")
    print("\nRequisitos previos:")
    print("1. Haber ejecutado download_model.py al menos una vez")
    print("2. Tener los archivos del modelo en ./models/")
    print("\n" + "=" * 50)
    
    confirm = input("¿Continuar? (s/n): ").strip().lower()
    
    if confirm == 's':
        set_offline_mode()
        print("\n🎯 Ahora puedes ejecutar: python main.py")
        print("   La aplicación usará los archivos locales")
    else:
        print("Operación cancelada")