"""
Provider de IA: Google Gemini
Versión simplificada compatible con múltiples versiones de google-generativeai
"""
from typing import Dict, Any
from .base_ai import BaseAI
from config import AIConfig

class GeminiAI(BaseAI):
    """
    Provider de Google Gemini para análisis de trading
    Versión simplificada sin dependencias complejas
    """
    
    def __init__(self):
        super().__init__(
            api_key=AIConfig.GEMINI_API_KEY,
            model=AIConfig.GEMINI_MODEL,
            max_tokens=1000
        )
        
        self.client = None
        self.genai = None
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.genai = genai
                self.client = True  # Marca como disponible
            except Exception as e:
                self.logger.error(f"Error configurando Gemini: {e}")
                self.client = None
    
    def get_provider_name(self) -> str:
        return "Gemini (Google)"
    
    def is_available(self) -> bool:
        return bool(self.api_key and self.client and self.api_key.startswith('AIza'))
    
    def analyze_market(self, message: str, market_data: Dict[str, Any]) -> str:
        """
        Análisis de mercado usando Gemini - Versión simplificada
        """
        if not self.is_available():
            return "❌ Google Gemini no está disponible. Verifica la API key en el archivo .env"
        
        try:
            # Formatear contexto
            market_context = self.format_market_context(market_data)
            
            # Crear prompt específico para Gemini
            prompt = self.create_gemini_prompt(message, market_context)
            
            # Intentar diferentes métodos de la API según la versión
            try:
                # Método 1: API moderna (v0.4+)
                model = self.genai.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                return response.text
            except AttributeError:
                try:
                    # Método 2: API intermedia (v0.3.x)
                    response = self.genai.generate_text(
                        model=self.model,
                        prompt=prompt,
                        temperature=0.7,
                        max_output_tokens=self.max_tokens
                    )
                    return response.result if hasattr(response, 'result') else str(response)
                except:
                    # Método 3: Fallback - Simulación local
                    return self._generate_fallback_analysis(message, market_context)
            
        except Exception as e:
            return self.handle_error(e, "Gemini API Error")
    
    def _generate_fallback_analysis(self, message: str, market_context: str) -> str:
        """
        Análisis de fallback cuando la API no funciona
        """
        return f"""## 🤖 Análisis de Gemini (Modo Fallback)

**Consulta:** {message}

**Contexto del mercado actual detectado en los datos:**
{market_context[:500]}...

### 📊 Análisis Técnico Simulado
Basado en los datos proporcionados, se observan las siguientes tendencias generales:

- **Momentum del mercado**: Mixto con volatilidad normal
- **Niveles de soporte**: En formación según datos históricos
- **Volumen**: Patrones típicos de trading institucional

### ⚠️ Gestión de Riesgo
- Mantener stop loss entre 3-8% según el asset
- Diversificación recomendada
- Monitorear correlaciones con BTC

### 🔧 Nota Técnica
*Gemini está funcionando en modo fallback debido a problemas de compatibilidad con la versión instalada de google-generativeai. Para funcionalidad completa:*

```bash
pip install --upgrade google-generativeai==0.4.0
```

**📋 Disclaimer:** Este análisis es educativo y no constituye consejo financiero.
        """
    
    def create_gemini_prompt(self, user_message: str, market_context: str) -> str:
        """
        Crea prompt simplificado para Gemini
        """
        return f"""Actúas como un analista cuantitativo especializado en criptomonedas.

DATOS DEL MERCADO:
{market_context}

CONSULTA: {user_message}

Proporciona un análisis estructurado en español con:
1. Resumen ejecutivo
2. Análisis técnico con niveles específicos  
3. Recomendaciones de gestión de riesgo
4. Acción sugerida

Formato markdown, máximo 400 palabras.
        """
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Capacidades de Gemini"""
        capabilities = super().get_capabilities()
        capabilities.update({
            'fast_inference': True,
            'cost_effective': True,
            'google_integration': True,
            'fallback_mode': True  # Capacidad única de este provider
        })
        return capabilities
    
    def get_strengths(self) -> list:
        """Fortalezas de Gemini para trading"""
        return [
            "⚡ Procesamiento ultra-rápido",
            "💰 Costo-efectivo para alto volumen",
            "🔄 Modo fallback cuando falla API",
            "🎯 Análisis conciso y directo",
            "🌐 Integración con ecosistema Google"
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