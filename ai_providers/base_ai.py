"""
Clase base abstracta para todos los providers de IA
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from datetime import datetime

class BaseAI(ABC):
    """Clase base para todos los providers de IA de trading"""
    
    def __init__(self, api_key: str, model: str, max_tokens: int = 1000):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def analyze_market(self, message: str, market_data: Dict[str, Any]) -> str:
        """
        Método principal para análisis de mercado
        Args:
            message: Pregunta/solicitud del usuario
            market_data: Datos actuales del mercado
        Returns:
            str: Análisis generado por la IA
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Retorna el nombre del provider"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Verifica si el provider está disponible (API key válida, etc.)"""
        pass
    
    def format_market_context(self, market_data: Dict[str, Any]) -> str:
        """
        Formatea los datos del mercado para el contexto de la IA
        """
        if not market_data:
            return "No hay datos de mercado disponibles."
            
        context = f"""
📊 DATOS ACTUALES DEL MERCADO ({datetime.now().strftime('%Y-%m-%d %H:%M')}):

"""
        
        # Precios principales
        if 'prices' in market_data:
            prices = market_data['prices']
            context += "💰 PRECIOS ACTUALES:\n"
            
            if 'bitcoin' in prices:
                btc = prices['bitcoin']
                context += f"• Bitcoin: ${btc['usd']:,.2f} ({btc.get('usd_24h_change', 0):+.2f}%)\n"
                
            if 'ethereum' in prices:
                eth = prices['ethereum'] 
                context += f"• Ethereum: ${eth['usd']:,.2f} ({eth.get('usd_24h_change', 0):+.2f}%)\n"
                
            if 'solana' in prices:
                sol = prices['solana']
                context += f"• Solana: ${sol['usd']:.2f} ({sol.get('usd_24h_change', 0):+.2f}%)\n"
        
        # Fear & Greed Index
        if 'fear_greed' in market_data:
            fg = market_data['fear_greed']
            context += f"\n😨 FEAR & GREED INDEX: {fg.get('value', 'N/A')}/100 ({fg.get('value_classification', 'N/A')})\n"
        
        # Datos globales del mercado
        if 'global_data' in market_data:
            global_data = market_data['global_data']
            market_cap = global_data.get('total_market_cap', {}).get('usd', 0)
            volume = global_data.get('total_volume', {}).get('usd', 0)
            btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0)
            
            context += f"""
🌐 MÉTRICAS GLOBALES:
• Market Cap Total: ${market_cap/1e12:.2f}T
• Volumen 24h: ${volume/1e9:.1f}B  
• Dominancia BTC: {btc_dominance:.1f}%
"""
        
        return context
    
    def create_trading_prompt(self, user_message: str, market_context: str) -> str:
        """
        Crea el prompt especializado para trading
        """
        return f"""Eres un analista financiero experto especializado en mercados de criptomonedas con más de 10 años de experiencia en trading institucional.

CONTEXTO DEL MERCADO:
{market_context}

ESPECIALIDADES:
• Análisis técnico avanzado (RSI, MACD, Bollinger Bands, Fibonacci)
• Análisis fundamental de criptomonedas  
• Identificación de soportes y resistencias
• Gestión de riesgos profesional
• Psicología del trading
• Estrategias de entrada y salida

INSTRUCCIONES:
1. Proporciona análisis concretos y actionables
2. Incluye niveles específicos de precio cuando sea relevante
3. Siempre menciona gestión de riesgos
4. Usa emojis para mejor legibilidad
5. Estructura la respuesta con headers en markdown
6. Incluye disclaimers apropiados

CONSULTA DEL TRADER:
{user_message}

RESPUESTA (en español, formato markdown):"""

    def handle_error(self, error: Exception, context: str = "") -> str:
        """
        Maneja errores de forma consistente
        """
        error_msg = f"❌ Error en {self.get_provider_name()}"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {str(error)}"
        
        self.logger.error(error_msg)
        
        return f"""
## ❌ Error de Conexión

Lo siento, hubo un problema conectando con {self.get_provider_name()}.

**Posibles causas:**
- Problema de conexión a internet
- API key inválida o expirada  
- Límite de rate limit excedido
- Mantenimiento del servicio

**Mientras tanto, puedes:**
- Verificar tu conexión
- Intentar con otro provider de IA
- Revisar los datos del mercado en la interfaz principal

*Error técnico: {str(error)}*
        """

    def get_capabilities(self) -> Dict[str, bool]:
        """
        Retorna las capacidades del provider
        """
        return {
            'market_analysis': True,
            'technical_indicators': True, 
            'risk_management': True,
            'real_time_data': True,
            'sentiment_analysis': True,
            'trading_strategies': True
        }