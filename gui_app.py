"""
Interfaz gráfica de DeepSeek Desktop
Usando CustomTkinter para mejor apariencia
Versión simplificada sin parámetros técnicos en UI
"""

import customtkinter as ctk
from tkinter import scrolledtext, messagebox
import threading
import time
import json
from pathlib import Path
from PIL import Image, ImageTk
import sys
import os

# Importar nuestro modelo
from deepseek_local import DeepSeekModel

class DeepSeekApp:
    """Aplicación principal con interfaz gráfica simplificada"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.is_generating = False
        self.current_stream = None
        
        # Parámetros fijos (ocultos al usuario)
        self.temperature = config.get("temperature", 0.7)
        self.max_length = config.get("max_length", 512)
        
        # Configurar apariencia
        ctk.set_appearance_mode(config.get("theme", "dark"))
        ctk.set_default_color_theme("blue")
        
        # Crear ventana principal
        self.root = ctk.CTk()
        self.root.title("DeepSeek Desktop - IA Local")
        self.root.geometry("900x700")
        
        # Cargar icono
        self._load_icon()
        
        # Variables de estado
        self.streaming_var = ctk.BooleanVar(value=True)
        
        # Construir interfaz
        self._create_widgets()
        
        # Cargar modelo en segundo plano
        self._load_model_async()
    
    def _load_icon(self):
        """Cargar o crear icono para la app"""
        try:
            # Intentar cargar icono desde archivo
            icon_path = Path("data/icon.png")
            if icon_path.exists():
                icon = ImageTk.PhotoImage(Image.open(icon_path))
                self.root.iconphoto(True, icon)
        except:
            pass
    
    def _create_widgets(self):
        """Crear todos los widgets de la interfaz"""
        
        # Frame principal
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        self.header_frame = ctk.CTkFrame(self.main_frame, height=80)
        self.header_frame.pack(fill="x", padx=5, pady=(5, 10))
        self.header_frame.pack_propagate(False)
        
        # Título
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="🤖 DeepSeek Desktop",
            font=("Arial", 24, "bold")
        )
        self.title_label.pack(side="left", padx=20, pady=10)
        
        # Estado del modelo
        self.status_label = ctk.CTkLabel(
            self.header_frame,
            text="Cargando modelo...",
            font=("Arial", 12),
            text_color="yellow"
        )
        self.status_label.pack(side="right", padx=20, pady=10)
        
        # Frame de contenido
        self.content_frame = ctk.CTkFrame(self.main_frame)
        self.content_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Panel izquierdo (chat)
        self.chat_frame = ctk.CTkFrame(self.content_frame)
        self.chat_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Área de chat
        self.chat_label = ctk.CTkLabel(
            self.chat_frame,
            text="💬 Conversación",
            font=("Arial", 14, "bold")
        )
        self.chat_label.pack(anchor="w", padx=10, pady=(10, 5))
        
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap="word",
            font=("Arial", 11),
            bg="#2b2b2b" if self.config.get("theme") == "dark" else "#ffffff",
            fg="#ffffff" if self.config.get("theme") == "dark" else "#000000",
            insertbackground="white",
            relief="flat",
            height=20
        )
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.chat_display.configure(state="disabled")
        
        # Entrada de texto
        self.input_frame = ctk.CTkFrame(self.chat_frame)
        self.input_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.input_text = ctk.CTkTextbox(
            self.input_frame,
            height=4,
            font=("Arial", 11)
        )
        self.input_text.pack(fill="x", padx=5, pady=5)
        self.input_text.bind("<Return>", self._on_enter_pressed)
        self.input_text.bind("<Shift-Return>", lambda e: "break")
        
        # Botones de chat
        self.button_frame = ctk.CTkFrame(self.input_frame)
        self.button_frame.pack(fill="x", padx=5, pady=(0, 5))
        
        self.send_button = ctk.CTkButton(
            self.button_frame,
            text="Enviar (Enter)",
            command=self.send_message,
            state="disabled"
        )
        self.send_button.pack(side="left", padx=(0, 5))
        
        self.clear_button = ctk.CTkButton(
            self.button_frame,
            text="Limpiar Chat",
            command=self.clear_chat,
            fg_color="gray30",
            hover_color="gray20"
        )
        self.clear_button.pack(side="left")
        
        # Panel derecho (configuración)
        self.sidebar_frame = ctk.CTkFrame(self.content_frame, width=250)
        self.sidebar_frame.pack(side="right", fill="y", padx=(5, 0))
        self.sidebar_frame.pack_propagate(False)
        
        # Información del modelo
        self.info_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="⚙️ Configuración",
            font=("Arial", 16, "bold")
        )
        self.info_label.pack(anchor="w", padx=15, pady=(15, 5))
        
        # Info del modelo
        self.model_info_text = ctk.CTkTextbox(
            self.sidebar_frame,
            height=150,
            font=("Arial", 10),
            state="disabled"
        )
        self.model_info_text.pack(fill="x", padx=15, pady=5)
        
        # Controles
        self.controls_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="📊 Opciones",
            font=("Arial", 14, "bold")
        )
        self.controls_label.pack(anchor="w", padx=15, pady=(15, 5))
        
        # Streaming
        self.stream_checkbox = ctk.CTkCheckBox(
            self.sidebar_frame,
            text="Streaming en tiempo real",
            variable=self.streaming_var,
            font=("Arial", 11)
        )
        self.stream_checkbox.pack(anchor="w", padx=15, pady=10)
        
        # Botones de acción
        self.action_frame = ctk.CTkFrame(self.sidebar_frame)
        self.action_frame.pack(fill="x", padx=15, pady=10)
        
        self.save_button = ctk.CTkButton(
            self.action_frame,
            text="💾 Guardar Chat",
            command=self.save_conversation,
            fg_color="green",
            hover_color="dark green"
        )
        self.save_button.pack(fill="x", pady=2)
        
        
        self.quit_button = ctk.CTkButton(
            self.action_frame,
            text="🚪 Salir",
            command=self.quit_app,
            fg_color="red",
            hover_color="dark red"
        )
        self.quit_button.pack(fill="x", pady=2)
        
        # Footer
        self.footer_frame = ctk.CTkFrame(self.main_frame, height=40)
        self.footer_frame.pack(fill="x", padx=5, pady=(10, 5))
        self.footer_frame.pack_propagate(False)
        
        self.footer_label = ctk.CTkLabel(
            self.footer_frame,
            text="Materia: Inteligencia Artificial - Proyecto DeepSeek Local",
            font=("Arial", 10)
        )
        self.footer_label.pack(side="left", padx=20, pady=10)
        
        self.token_label = ctk.CTkLabel(
            self.footer_frame,
            text="Tokens: 0",
            font=("Arial", 10)
        )
        self.token_label.pack(side="right", padx=20, pady=10)
    
    def _load_model_async(self):
        """Cargar modelo en segundo plano"""
        def load_model():
            try:
                self.model = DeepSeekModel(
                    model_name=self.config["model"],
                    device=self.config["device"],
                    use_quantization=self.config.get("quantization") == "8bit"
                )
                
                # Actualizar UI en hilo principal
                self.root.after(0, self._on_model_loaded)
                
            except Exception as e:
                self.root.after(0, lambda: self._on_model_error(str(e)))
        
        # Iniciar hilo
        thread = threading.Thread(target=load_model, daemon=True)
        thread.start()
    
    def _on_model_loaded(self):
        """Cuando el modelo se carga exitosamente"""
        self.status_label.configure(
            text="✅ Modelo cargado",
            text_color="light green"
        )
        self.send_button.configure(state="normal")
        
        # Actualizar información del modelo
        info = self.model.get_model_info()
        info_text = f"""Modelo: {self.config['model']}
Parámetros: ~{info['parameters']:,}
Dispositivo: {info['device'].upper()}
Tiempo carga: {info['load_time']:.1f}s

Configuración:
• Temperatura: {self.temperature}
• Longitud: {self.max_length} tokens
• Streaming: {'Sí' if self.streaming_var.get() else 'No'}"""
        
        self.model_info_text.configure(state="normal")
        self.model_info_text.delete("1.0", "end")
        self.model_info_text.insert("1.0", info_text)
        self.model_info_text.configure(state="disabled")
        
        # Mensaje de bienvenida
        self._add_to_chat("Sistema", f"Modelo '{self.config['model']}' cargado exitosamente.")
        self._add_to_chat("Sistema", "¡Escribe tu mensaje abajo y presiona Enter!")
    
    def _on_model_error(self, error_msg):
        """Cuando hay error cargando el modelo"""
        self.status_label.configure(
            text="❌ Error cargando modelo",
            text_color="red"
        )
        messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{error_msg}")
    
    def _on_enter_pressed(self, event):
        """Manejar Enter para enviar mensaje"""
        if not event.state & 0x1:  # Si no está presionado Shift
            self.send_message()
            return "break"  # Prevenir nueva línea
        return None
    
    def send_message(self):
        """Enviar mensaje al modelo"""
        if self.is_generating or self.model is None:
            return
        
        # Obtener texto
        prompt = self.input_text.get("1.0", "end-1c").strip()
        if not prompt:
            return
        
        # Limpiar entrada
        self.input_text.delete("1.0", "end")
        
        # Deshabilitar controles durante generación
        self.is_generating = True
        self.send_button.configure(state="disabled")
        self.input_text.configure(state="disabled")
        
        # Mostrar mensaje del usuario
        self._add_to_chat("Tú", prompt)
        
        # Generar respuesta en segundo plano
        if self.streaming_var.get():
            threading.Thread(target=self._generate_streaming, args=(prompt,), daemon=True).start()
        else:
            threading.Thread(target=self._generate_normal, args=(prompt,), daemon=True).start()
    
    def _generate_normal(self, prompt):
        """Generación normal (sin streaming)"""
        try:
            response = self.model.generate(
                prompt=prompt,
                max_length=self.max_length,
                temperature=self.temperature
            )
            
            # Mostrar respuesta en UI
            self.root.after(0, lambda: self._add_to_chat("DeepSeek", response))
            
        except Exception as e:
            self.root.after(0, lambda: self._add_to_chat("Error", str(e)))
        
        finally:
            self.root.after(0, self._enable_input)
    
    def _generate_streaming(self, prompt):
        """Generación con streaming"""
        try:
            # Crear área para respuesta streaming
            self.root.after(0, lambda: self._start_streaming_response())
            
            # Generar token por token
            full_response = ""
            for token in self.model.generate_streaming(
                prompt=prompt,
                max_length=self.max_length,
                temperature=self.temperature
            ):
                full_response += token
                self.root.after(0, lambda t=token: self._append_to_streaming(t))
            
            # Guardar respuesta completa
            if self.model:
                self.model._save_to_history(prompt, full_response)
            
        except Exception as e:
            self.root.after(0, lambda: self._add_to_chat("Error", str(e)))
        
        finally:
            self.root.after(0, self._enable_input)
            self.root.after(0, self._end_streaming_response)
    
    def _start_streaming_response(self):
        """Iniciar respuesta con streaming"""
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", "\n🤖 DeepSeek: ", "assistant")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")
    
    def _append_to_streaming(self, token):
        """Añadir token a respuesta streaming"""
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", token)
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")
    
    def _end_streaming_response(self):
        """Finalizar respuesta streaming"""
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", "\n\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")
    
    def _enable_input(self):
        """Habilitar entrada después de generación"""
        self.is_generating = False
        self.send_button.configure(state="normal")
        self.input_text.configure(state="normal")
        self.input_text.focus()
    
    def _add_to_chat(self, sender, message):
        """Añadir mensaje al chat"""
        self.chat_display.configure(state="normal")
        
        # Añadir separador si no es el primer mensaje
        if self.chat_display.get("1.0", "end-1c").strip():
            self.chat_display.insert("end", "\n" + "─" * 50 + "\n\n")
        
        # Añadir mensaje
        if sender == "Tú":
            self.chat_display.insert("end", "👤 Tú:\n", "user")
            self.chat_display.insert("end", f"{message}\n", "user_text")
        elif sender == "DeepSeek":
            self.chat_display.insert("end", "🤖 DeepSeek:\n", "assistant")
            self.chat_display.insert("end", f"{message}\n", "assistant_text")
        else:
            self.chat_display.insert("end", f"⚙️ {sender}:\n{message}\n")
        
        # Configurar tags para colores
        self.chat_display.tag_config("user", foreground="#4fc3f7")
        self.chat_display.tag_config("user_text", foreground="#ffffff")
        self.chat_display.tag_config("assistant", foreground="#69f0ae")
        self.chat_display.tag_config("assistant_text", foreground="#e0e0e0")
        
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")
        
        # Actualizar contador de tokens
        tokens = len(message.split()) * 1.3  # Estimación aproximada
        self.token_label.configure(text=f"Tokens: ~{int(tokens)}")
    
    def clear_chat(self):
        """Limpiar el chat"""
        if messagebox.askyesno("Limpiar chat", "¿Estás seguro de querer limpiar el chat?"):
            self.chat_display.configure(state="normal")
            self.chat_display.delete("1.0", "end")
            self.chat_display.configure(state="disabled")
            
            if self.model:
                self.model.clear_history()
            
            self._add_to_chat("Sistema", "Chat limpiado.")
    
    def save_conversation(self):
        """Guardar conversación actual"""
        try:
            chat_content = self.chat_display.get("1.0", "end-1c")
            if not chat_content.strip():
                messagebox.showwarning("Advertencia", "No hay conversación para guardar.")
                return
            
            # Crear nombre de archivo con timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/conversations/chat_{timestamp}.txt"
            
            # Guardar
            Path("data/conversations").mkdir(parents=True, exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(chat_content)
            
            messagebox.showinfo("Guardado", f"Conversación guardada en:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar: {str(e)}")
    
    def load_conversation(self):
        """Cargar conversación anterior"""
        # Implementación simplificada - en versión completa usar filedialog
        messagebox.showinfo("Cargar", "Función de carga en desarrollo.")
    
    def export_conversation(self):
        """Exportar conversación a formato legible"""
        self.save_conversation()  # Por ahora, misma funcionalidad
    
    def quit_app(self):
        """Salir de la aplicación"""
        if messagebox.askyesno("Salir", "¿Estás seguro de querer salir?"):
            # Guardar configuración
            self.config["temperature"] = self.temperature
            self.config["max_length"] = self.max_length
            
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            self.root.destroy()
            sys.exit(0)
    
    def run(self):
        """Ejecutar aplicación"""
        self.root.mainloop()