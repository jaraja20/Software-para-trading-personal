"""
Provider de IA: Anthropic Claude
Especializado en análisis financiero profundo y razonamiento complejo
"""
from typing import Dict, Any
import requests
import json
from .base_ai import BaseAI
from config import AIConfig

class ClaudeAI(BaseAI):
    """
    Provider de Claude/Anthropic para análisis de trading
    Fortalezas: Análisis contextual profundo, razonamiento lógico, análisis de documentos
    """
    
    def __init__(self):
        super().__init__(
            api_key=AIConfig.ANTHROPIC_API_KEY,
            model=AIConfig.CLAUDE_MODEL,
            max_tokens=AIConfig.CLAUDE_MAX_TOKENS
        )
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    def get_provider_name(self) -> str:
        return "Claude (Anthropic)"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def analyze_market(self, message: str, market_data: Dict[str, Any]) -> str:
        """
        Análisis de mercado usando Claude
        """
        if not self.is_available():
            return "❌ Claude AI no está disponible. Verifica la API key en el archivo .env"
        
        try:
            # Formatear contexto
            market_context = self.format_market_context(market_data)
            
            # Crear prompt especializado para Claude
            prompt = self.create_claude_prompt(message, market_context)
            
            # Hacer request a la API
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=AIConfig.AI_REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["content"][0]["text"]
            else:
                error_details = response.text
                return self.handle_error(
                    Exception(f"API Error {response.status_code}: {error_details}"),
                    "API Request"
                )
                
        except requests.RequestException as e:
            return self.handle_error(e, "Network Error")
        except Exception as e:
            return self.handle_error(e, "General Error")
    
    def create_claude_prompt(self, user_message: str, market_context: str) -> str:
        """
        Crea prompt específico optimizado para Claude
        """
        return f"""Eres Claude, un analista financiero senior especializado en criptomonedas con experiencia institucional en Goldman Sachs y Citadel.

<market_data>
{market_context}
</market_data>

<user_query>
{user_message}
</user_query>

<instructions>
Como experto en trading cuantitativo, proporciona un análisis detallado siguiendo esta estructura:

1. RESUMEN EJECUTIVO (2-3 líneas clave)
2. ANÁLISIS TÉCNICO (niveles específicos, indicadores)  
3. ANÁLISIS FUNDAMENTAL (catalyzadores, noticias relevantes)
4. GESTIÓN DE RIESGOS (stops, sizing, R:R ratios)
5. ACCIÓN RECOMENDADA (entrada/salida/esperar)

CRITERIOS IMPORTANTES:
- Incluye precios/niveles específicos cuando sea relevante
- Menciona timeframes apropiados
- Considera volatilidad histórica de cada asset
- Siempre incluye gestión de riesgo
- Usa formato markdown con emojis para legibilidad
- Sé conciso pero comprehensivo
- Incluye disclaimers apropiados

ESTILO: Profesional pero accesible, como briefing para trader institucional
</instructions>

Análisis:"""
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Capacidades específicas de Claude"""
        capabilities = super().get_capabilities()
        capabilities.update({
            'deep_reasoning': True,
            'document_analysis': True, 
            'contextual_understanding': True,
            'risk_assessment': True,
            'long_form_analysis': True
        })
        return capabilities
    
    def get_strengths(self) -> list:
        """Fortalezas específicas de Claude para trading"""
        return [
            "📊 Análisis contextual profundo",
            "🧠 Razonamiento lógico superior", 
            "📈 Interpretación de datos complejos",
            "⚖️ Evaluación de riesgos avanzada",
            "📋 Análisis de documentos financieros",
            "🎯 Recomendaciones específicas y actionables"
        ]