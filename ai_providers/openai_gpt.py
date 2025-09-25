"""
Provider de IA: OpenAI GPT-4
Especializado en análisis versátil y respuestas creativas
"""
from typing import Dict, Any
import openai
from .base_ai import BaseAI
from config import AIConfig

class OpenAIGPT(BaseAI):
    """
    Provider de OpenAI GPT-4 para análisis de trading
    Fortalezas: Versatilidad, creatividad, gran ecosistema, respuestas balanceadas
    """
    
    def __init__(self):
        super().__init__(
            api_key=AIConfig.OPENAI_API_KEY,
            model=AIConfig.OPENAI_MODEL,
            max_tokens=AIConfig.OPENAI_MAX_TOKENS
        )
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
        self.temperature = AIConfig.OPENAI_TEMPERATURE
    
    def get_provider_name(self) -> str:
        return "GPT-4 (OpenAI)"
    
    def is_available(self) -> bool:
        return bool(self.api_key and self.client)
    
    def analyze_market(self, message: str, market_data: Dict[str, Any]) -> str:
        """
        Análisis de mercado usando GPT-4
        """
        if not self.is_available():
            return "❌ OpenAI GPT-4 no está disponible. Verifica la API key en el archivo .env"
        
        try:
            # Formatear contexto
            market_context = self.format_market_context(market_data)
            
            # Crear prompt específico para GPT-4
            prompt = self.create_gpt_prompt(message, market_context)
            
            # Hacer request a la API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": self.get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=AIConfig.AI_REQUEST_TIMEOUT
            )
            
            return response.choices[0].message.content
            
        except openai.APIError as e:
            return self.handle_error(e, f"OpenAI API Error: {e.code}")
        except openai.RateLimitError as e:
            return self.handle_error(e, "Rate Limit Exceeded")
        except openai.APIConnectionError as e:
            return self.handle_error(e, "Connection Error") 
        except Exception as e:
            return self.handle_error(e, "General Error")
    
    def get_system_prompt(self) -> str:
        """
        System prompt optimizado para GPT-4 en trading
        """
        return """Eres un analista financiero senior con 15 años de experiencia en Wall Street, especializado en:

🏦 EXPERIENCIA:
- Ex-trader en JPMorgan y BlackRock
- Especialista en mercados de criptomonedas desde 2017
- Certificado CFA y FRM
- Experiencia en trading algorítmico y análisis cuantitativo

🎯 METODOLOGÍA:
- Combinación de análisis técnico y fundamental
- Enfoque en gestión de riesgos disciplinada  
- Uso de métricas cuantitativas
- Consideración de factores macro y micro

📊 HERRAMIENTAS:
- Indicadores técnicos (RSI, MACD, Bollinger, Fibonacci)
- Análisis de volumen y flujo de órdenes
- Sentiment analysis y positioning
- Risk/reward ratios

🎨 ESTILO DE COMUNICACIÓN:
- Profesional pero accesible
- Datos concretos y actionables
- Estructura clara con markdown
- Emojis para mejor legibilidad
- Siempre incluir disclaimers

Tu objetivo es proporcionar análisis de trading de calidad institucional."""
    
    def create_gpt_prompt(self, user_message: str, market_context: str) -> str:
        """
        Crea prompt específico para GPT-4
        """
        return f"""DATOS ACTUALES DEL MERCADO:
{market_context}

CONSULTA DEL TRADER:
{user_message}

INSTRUCCIONES ESPECÍFICAS:
1. Analiza la consulta en el contexto de los datos actuales
2. Proporciona niveles de precio específicos cuando sea relevante
3. Incluye análisis de riesgo/recompensa
4. Sugiere timeframes apropiados
5. Menciona gestión de posición y stops
6. Usa estructura markdown clara
7. Incluye emojis para organización visual

FORMATO DE RESPUESTA:
## [Título del Análisis]
### 📊 Resumen Ejecutivo
### 📈 Análisis Técnico  
### 🎯 Recomendación
### ⚠️ Gestión de Riesgos
### 💡 Consideraciones Adicionales

Responde en español, de forma profesional pero accesible:"""
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Capacidades específicas de GPT-4"""
        capabilities = super().get_capabilities()
        capabilities.update({
            'creative_analysis': True,
            'multi_modal': True,
            'plugin_support': True,
            'broad_knowledge': True,
            'balanced_responses': True,
            'code_generation': True
        })
        return capabilities
    
    def get_strengths(self) -> list:
        """Fortalezas específicas de GPT-4 para trading"""
        return [
            "🎯 Análisis versátil y balanceado",
            "💡 Creatividad en estrategias",
            "🔗 Gran ecosistema y herramientas", 
            "📚 Amplio conocimiento general",
            "⚡ Respuestas rápidas y consistentes",
            "🛠️ Capacidad de generar código/scripts"
        ]