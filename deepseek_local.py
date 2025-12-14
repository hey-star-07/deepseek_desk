"""
Módulo principal del modelo DeepSeek
Versión optimizada para OFFLINE con caché local
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Generator
import time
import json
import os
from datetime import datetime
from pathlib import Path

class DeepSeekModel:
    """Clase principal del modelo DeepSeek con soporte offline"""
    
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct",
                 device: str = "cpu", use_quantization: bool = False,
                 local_cache_path: str = "./models", offline_mode: bool = True):
        """
        Inicializa el modelo DeepSeek con caché local
        
        Args:
            model_name: Nombre del modelo en HuggingFace
            device: 'cpu' o 'cuda'
            use_quantization: Usar cuantización 8-bit
            local_cache_path: Ruta al caché local
            offline_mode: Modo sin conexión a internet
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.use_quantization = use_quantization
        self.local_cache_path = Path(local_cache_path)
        self.offline_mode = offline_mode
        self.model = None
        self.tokenizer = None
        self.history = []
        self.load_time = None
        
        print(f"🔄 Inicializando modelo: {model_name}")
        print(f"   Dispositivo: {self.device}")
        print(f"   Quantization: {use_quantization}")
        print(f"   Modo Offline: {offline_mode}")
        
        # Crear directorio de caché si no existe
        self.local_cache_path.mkdir(parents=True, exist_ok=True)
        
        self._load_model()
    
    def _find_local_model(self):
        """Busca archivos del modelo en caché local"""
        model_files = {
            "config": None,
            "tokenizer_config": None,
            "generation_config": None,
            "model": None,
            "tokenizer": None
        }
        
        # Patrones de archivos a buscar
        patterns = {
            "config": ["config.json", "*/config.json"],
            "tokenizer_config": ["tokenizer_config.json", "*/tokenizer_config.json"],
            "generation_config": ["generation_config.json", "*/generation_config.json"],
            "model": ["*.safetensors", "pytorch_model.bin", "*.bin"],
            "tokenizer": ["tokenizer.json", "vocab.json", "merges.txt", "vocab.txt"]
        }
        
        for key, pattern_list in patterns.items():
            for pattern in pattern_list:
                files = list(self.local_cache_path.glob(pattern))
                if files:
                    model_files[key] = str(files[0])
                    break
        
        # Verificar si tenemos los archivos esenciales
        essential_files = ["config", "model", "tokenizer"]
        missing_essential = [f for f in essential_files if model_files[f] is None]
        
        return model_files, missing_essential
    
    def _download_if_needed(self):
        """Descarga el modelo si no existe localmente"""
        model_files, missing = self._find_local_model()
        
        if not missing:
            print("✅ Modelo encontrado en caché local")
            return True
        
        if self.offline_mode:
            print("❌ Modo offline activado - no se puede descargar")
            print(f"   Archivos faltantes: {', '.join(missing)}")
            return False
        
        print("📥 Algunos archivos faltan, descargando...")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Descargar tokenizador
            print("   Descargando tokenizador...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.local_cache_path),
                local_files_only=False,
                trust_remote_code=True
            )
            
            # Descargar modelo
            print("   Descargando modelo (puede tomar tiempo)...")
            load_kwargs = {
                "cache_dir": str(self.local_cache_path),
                "local_files_only": False,
                "trust_remote_code": True,
                "torch_dtype": torch.float32 if self.device == "cpu" else torch.float16
            }
            
            if self.use_quantization and self.device == "cpu":
                load_kwargs["load_in_8bit"] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error descargando: {str(e)}")
            return False
    
    def _load_from_local(self):
        """Carga el modelo desde caché local"""
        try:
            # Configurar variables de entorno para forzar modo local
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            # Buscar archivos locales
            model_files, missing = self._find_local_model()
            
            if not model_files["config"] or not model_files["model"]:
                raise FileNotFoundError("Archivos esenciales del modelo no encontrados")
            
            # Cargar desde archivos específicos
            print("📂 Cargando desde caché local...")
            
            # Cargar configuración
            config_path = model_files["config"]
            print(f"   Config: {config_path}")
            
            # Cargar tokenizador
            if model_files["tokenizer_config"]:
                tokenizer_path = str(Path(model_files["tokenizer_config"]).parent)
                print(f"   Tokenizer: {tokenizer_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
            else:
                # Intentar cargar con el directorio base
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.local_cache_path),
                    local_files_only=True,
                    trust_remote_code=True
                )
            
            # Configurar padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Cargar modelo
            print("   Cargando pesos del modelo...")
            load_kwargs = {
                "local_files_only": True,
                "trust_remote_code": True,
                "torch_dtype": torch.float32 if self.device == "cpu" else torch.float16
            }
            
            if self.use_quantization and self.device == "cpu":
                load_kwargs["load_in_8bit"] = True
                print("   Usando quantización 8-bit")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.local_cache_path),
                **load_kwargs
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando desde local: {str(e)}")
            return False
    
    def _load_model(self):
        """Carga el modelo usando caché local"""
        start_time = time.time()
        
        try:
            # Primero intentar cargar desde caché local
            success = self._load_from_local()
            
            if not success and not self.offline_mode:
                # Si falla y no estamos en modo offline, intentar descargar
                print("🔄 Intentando descargar modelo...")
                success = self._download_if_needed()
            
            if success:
                # Mover a dispositivo
                if self.device == "cpu":
                    self.model = self.model.to(torch.float32)
                self.model = self.model.to(self.device)
                self.model.eval()
                
                self.load_time = time.time() - start_time
                print(f"✅ Modelo cargado en {self.load_time:.2f} segundos")
                print(f"   Parámetros: ~{self._count_parameters():,}")
                
                # Guardar información del modelo
                self._save_model_info()
            else:
                raise RuntimeError("No se pudo cargar el modelo desde caché local")
                
        except Exception as e:
            print(f"❌ Error cargando modelo: {str(e)}")
            
            # Crear modelo dummy para pruebas si no hay conexión
            if self.offline_mode:
                print("⚠️  Creando modelo dummy para demostración...")
                self._create_dummy_model()
            else:
                raise
    
    def _create_dummy_model(self):
        """Crea un modelo dummy para demostración sin internet"""
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        
        print("🧪 Usando modelo dummy GPT-2 (sin conexión requerida)")
        
        # Usar GPT-2 pequeño como fallback
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.model_name = "gpt2 (dummy - sin conexión)"
        self.load_time = 0.1
    
    def _save_model_info(self):
        """Guarda información del modelo cargado"""
        info = {
            "model_name": self.model_name,
            "loaded_at": datetime.now().isoformat(),
            "device": self.device,
            "parameters": self._count_parameters(),
            "quantization": self.use_quantization,
            "cache_path": str(self.local_cache_path)
        }
        
        info_path = self.local_cache_path / "model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    
    def _count_parameters(self) -> int:
        """Cuenta parámetros del modelo"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def generate(self, prompt: str, max_length: int = 512, 
                temperature: float = 0.7, top_p: float = 0.9,
                system_message: str = None) -> str:
        """
        Genera respuesta a partir de un prompt
        
        Args:
            prompt: Texto de entrada
            max_length: Longitud máxima
            temperature: Creatividad (0.1-1.0)
            top_p: Nucleus sampling
            system_message: Mensaje del sistema
            
        Returns:
            Respuesta generada
        """
        if self.model is None or self.tokenizer is None:
            return "Error: Modelo no cargado"
        
        try:
            # Formatear prompt
            formatted_prompt = self._format_prompt(prompt, system_message)
            
            # Tokenizar
            inputs = self.tokenizer.encode(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Configurar generación
            generation_config = {
                "max_new_tokens": min(max_length, 1024),
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1
            }
            
            # Generar
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    **generation_config
                )
            
            # Decodificar
            response = self.tokenizer.decode(
                outputs[0][len(inputs[0]):],
                skip_special_tokens=True
            )
            
            # Guardar en historial
            self._save_to_history(prompt, response)
            
            return response.strip()
            
        except Exception as e:
            return f"Error en generación: {str(e)}"
    
    def generate_streaming(self, prompt: str, max_length: int = 512,
                          temperature: float = 0.7) -> Generator[str, None, None]:
        """
        Genera respuesta con streaming (token por token)
        
        Args:
            prompt: Texto de entrada
            max_length: Longitud máxima
            temperature: Creatividad
            
        Yields:
            Tokens generados uno por uno
        """
        if self.model is None or self.tokenizer is None:
            yield "Error: Modelo no cargado"
            return
        
        try:
            # Formatear prompt
            formatted_prompt = self._format_prompt(prompt)
            
            # Tokenizar
            inputs = self.tokenizer.encode(
                formatted_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Generar token por token
            generated = inputs.clone()
            attention_mask = torch.ones_like(inputs)
            
            for _ in range(max_length):
                with torch.no_grad():
                    outputs = self.model(
                        generated,
                        attention_mask=attention_mask
                    )
                    
                    # Obtener siguiente token
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Si es token de fin, terminar
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Añadir token y actualizar máscara
                generated = torch.cat([generated, next_token], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=self.device)], 
                    dim=-1
                )
                
                # Decodificar y yield
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                yield token_text
                
                # Pequeña pausa para efecto visual
                time.sleep(0.01)
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def _format_prompt(self, prompt: str, system_message: str = None) -> str:
        """Formatea el prompt según el modelo"""
        if system_message is None:
            system_message = "Eres DeepSeek, un asistente de IA útil y amigable."
        
        # Si es modelo dummy (GPT-2)
        if "gpt2" in self.model_name.lower():
            return f"{prompt}\n\nAssistant:"
        
        # Formato para modelos DeepSeek-Coder
        elif "coder" in self.model_name.lower():
            return f"""### System Message:
{system_message}

### User Message:
{prompt}

### Assistant:
"""
        # Formato general
        else:
            return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
    
    def _save_to_history(self, prompt: str, response: str):
        """Guarda conversación en historial"""
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "model": self.model_name
        }
        
        self.history.append(conversation)
        
        # Guardar en archivo
        history_file = Path("data/conversations/history.json")
        if not history_file.parent.exists():
            history_file.parent.mkdir(parents=True)
        
        # Cargar historial existente
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    all_history = json.load(f)
            except:
                all_history = []
        else:
            all_history = []
        
        # Añadir nuevo y guardar
        all_history.append(conversation)
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(all_history[-100:], f, indent=2, ensure_ascii=False)
    
    def get_history(self) -> List[Dict]:
        """Obtiene historial de conversaciones"""
        return self.history[-10:]  # Últimas 10 conversaciones
    
    def clear_history(self):
        """Limpia el historial"""
        self.history.clear()
    
    def get_model_info(self) -> Dict:
        """Obtiene información del modelo"""
        return {
            "name": self.model_name,
            "device": self.device,
            "parameters": self._count_parameters(),
            "quantization": self.use_quantization,
            "load_time": self.load_time,
            "history_count": len(self.history),
            "cache_path": str(self.local_cache_path),
            "offline_mode": self.offline_mode
        }