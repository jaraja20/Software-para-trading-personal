"""
Provider de IA: Google Gemini
Especializado en análisis multimodal y datos en tiempo real
"""
from typing import Dict, Any
import google.generativeai as genai
from .base_ai import BaseAI
from config import AIConfig

class GeminiAI(BaseAI):
    """
    Provider de Google Gemini para análisis de trading
    Fortalezas: Análisis multimodal, datos en tiempo real, integración Google
    """
    
    def __init__(self):
        super().__init__(
            api_key=AIConfig.GEMINI_API_KEY,
            model=AIConfig.GEMINI_MODEL,
            max_tokens=1000  # Gemini maneja tokens diferente
        )
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
        else:
            self.client = None
    
    def get_provider_name(self) -> str:
        return "Gemini (Google)"
    
    def is_available(self) -> bool:
        return bool(self.api_key and self.client)
    
    def analyze_market(self, message: str, market_data: Dict[str, Any]) -> str:
        """
        Análisis de mercado usando Gemini
        """
        if not self.is_available():
            return "❌ Google Gemini no está disponible. Verifica la API key en el archivo .env"
        
        try:
            # Formatear contexto
            market_context = self.format_market_context(market_data)
            
            # Crear prompt específico para Gemini
            prompt = self.create_gemini_prompt(message, market_context)
            
            # Configurar parámetros de generación
            generation_config = {
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": self.max_tokens,
            }
            
            # Configuración de seguridad (permite contenido financiero)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            # Generar respuesta
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            return response.text
            
        except Exception as e:
            return self.handle_error(e, "Gemini API Error")
    
    def create_gemini_prompt(self, user_message: str, market_context: str) -> str:
        """
        Crea prompt específico optimizado para Gemini
        """
        return f"""Actúas como un analista cuantitativo senior de Google DeepMind especializado en criptomonedas, con acceso a datos en tiempo real y capacidades avanzadas de análisis multimodal.

CONTEXTO DE MERCADO ACTUAL:
{market_context}

CONSULTA DEL TRADER:
{user_message}

CAPACIDADES ÚNICAS DE GEMINI:
- Análisis de patrones complejos en grandes datasets
- Procesamiento multimodal (texto, números, gráficos conceptuales)
- Acceso a información actualizada de Google
- Análisis de correlaciones entre múltiples variables

ESTRUCTURA DE RESPUESTA REQUERIDA:

## 🎯 ANÁLISIS [ASSET/TEMA]

### 📊 DATOS CLAVE
- Precios actuales y momentum
- Volumen y actividad
- Métricas relevantes

### 🔍 ANÁLISIS CUANTITATIVO  
- Indicadores técnicos calculados
- Correlaciones identificadas
- Patrones históricos similares

### 🌐 CONTEXTO MACRO
- Factores externos influyentes
- Sentiment del mercado
- Noticias/eventos relevantes

### 🎯 RECOMENDACIÓN TÁCTICA
- Acción específica sugerida
- Niveles de precio clave
- Timeframes recomendados

### ⚠️ GESTIÓN DE RIESGOS
- Stops sugeridos
- Sizing de posición
- Escenarios alternativos

INSTRUCCIONES:
- Usa datos específicos del contexto proporcionado
- Incluye cálculos cuando sea relevante
- Considera múltiples timeframes
- Formato markdown con emojis
- Respuesta en español
- Máximo 800 palabras, información densa

Análisis:"""
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Capacidades específicas de Gemini"""
        capabilities = super().get_capabilities()
        capabilities.update({
            'multimodal_analysis': True,
            'real_time_data': True,
            'pattern_recognition': True,
            'google_integration': True,
            'large_context_window': True,
            'fast_inference': True
        })
        return capabilities
    
    def get_strengths(self) -> list:
        """Fortalezas específicas de Gemini para trading"""
        return [
            "🔍 Análisis multimodal avanzado",
            "⚡ Procesamiento ultra-rápido",
            "🌐 Acceso a datos de Google",
            "📊 Pattern recognition superior",
            "🔗 Integración con Google Workspace",
            "💰 Costo-efectivo para alto volumen"
        ]
    
    def analyze_chart_pattern(self, chart_description: str, market_data: Dict[str, Any]) -> str:
        """
        Análisis específico de patrones de gráfico (capacidad única de Gemini)
        """
        prompt = f"""Como experto en análisis técnico, identifica patrones en esta descripción de gráfico:

DESCRIPCIÓN DEL GRÁFICO:
{chart_description}

DATOS DE MERCADO:
{self.format_market_context(market_data)}

Identifica:
1. Patrones técnicos clásicos (triángulos, banderas, cabeza y hombros, etc.)
2. Niveles de soporte/resistencia
3. Volumen patterns
4. Confluencias técnicas
5. Probabilidades de continuación vs reversión

Respuesta estructurada en español con niveles específicos de precio."""
        
        try:
            if self.client:
                response = self.client.generate_content(prompt)
                return response.text
            else:
                return "❌ Gemini no disponible para análisis de patrones"
        except Exception as e:
            return self.handle_error(e, "Pattern Analysis Error")