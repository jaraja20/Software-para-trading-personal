"""
Gestor centralizado de providers de IA para Trading Assistant
"""
from typing import Dict, List, Optional, Any
import logging
from .base_ai import BaseAI
from .claude_ai import ClaudeAI
from .openai_gpt import OpenAIGPT
from .gemini_ai import GeminiAI
from config import AIConfig

class AIProviderManager:
    """
    Gestor centralizado para todos los providers de IA
    """
    
    def __init__(self):
        self.providers: Dict[str, BaseAI] = {}
        self.logger = logging.getLogger(__name__)
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Inicializa todos los providers disponibles"""
        provider_classes = {
            'claude': ClaudeAI,
            'openai': OpenAIGPT,
            'gemini': GeminiAI
        }
        
        for name, provider_class in provider_classes.items():
            try:
                provider = provider_class()
                if provider.is_available():
                    self.providers[name] = provider
                    self.logger.info(f"✅ {provider.get_provider_name()} inicializado correctamente")
                else:
                    self.logger.warning(f"⚠️ {name} no disponible (falta API key)")
            except Exception as e:
                self.logger.error(f"❌ Error inicializando {name}: {e}")
    
    def get_available_providers(self) -> List[str]:
        """Retorna lista de providers disponibles"""
        return list(self.providers.keys())
    
    def get_provider(self, provider_name: str) -> Optional[BaseAI]:
        """Obtiene un provider específico"""
        return self.providers.get(provider_name)
    
    def get_default_provider(self) -> Optional[BaseAI]:
        """Obtiene el provider por defecto"""
        default_name = AIConfig.DEFAULT_AI_PROVIDER
        provider = self.providers.get(default_name)
        
        if not provider and self.providers:
            # Si el default no está disponible, usar el primero disponible
            provider = next(iter(self.providers.values()))
            self.logger.warning(f"Provider por defecto '{default_name}' no disponible. Usando {provider.get_provider_name()}")
        
        return provider
    
    def analyze_with_provider(self, provider_name: str, message: str, market_data: Dict[str, Any]) -> str:
        """
        Analiza usando un provider específico
        """
        provider = self.get_provider(provider_name)
        if not provider:
            available = ", ".join(self.get_available_providers())
            return f"❌ Provider '{provider_name}' no disponible. Providers disponibles: {available}"
        
        try:
            return provider.analyze_market(message, market_data)
        except Exception as e:
            self.logger.error(f"Error en análisis con {provider_name}: {e}")
            return provider.handle_error(e, "Analysis Error")
    
    def analyze_with_default(self, message: str, market_data: Dict[str, Any]) -> str:
        """
        Analiza usando el provider por defecto
        """
        provider = self.get_default_provider()
        if not provider:
            return """❌ **No hay providers de IA disponibles**
            
**Para habilitar IA, agrega al menos una API key en el archivo .env:**

```bash
# OpenAI (GPT-4)
OPENAI_API_KEY=sk-your-key-here

# Anthropic (Claude)  
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Google (Gemini)
GEMINI_API_KEY=your-gemini-key-here
```

**Mientras tanto, puedes:**
- Usar los datos del mercado en tiempo real
- Analizar los gráficos de velas
- Consultar métricas como Fear & Greed Index
            """
        
        return provider.analyze_market(message, market_data)
    
    def get_provider_comparison(self) -> str:
        """
        Genera una comparación de todos los providers disponibles
        """
        if not self.providers:
            return "❌ No hay providers de IA disponibles"
        
        comparison = "## 🤖 PROVIDERS DE IA DISPONIBLES\n\n"
        
        for name, provider in self.providers.items():
            comparison += f"### {provider.get_provider_name()}\n"
            comparison += f"**Estado:** ✅ Disponible\n"
            comparison += f"**Fortalezas:**\n"
            
            for strength in provider.get_strengths():
                comparison += f"- {strength}\n"
            
            capabilities = provider.get_capabilities()
            special_caps = [k for k, v in capabilities.items() if v and k not in ['market_analysis', 'technical_indicators', 'risk_management']]
            
            if special_caps:
                comparison += f"**Capacidades especiales:** {', '.join(special_caps)}\n"
            
            comparison += "\n"
        
        comparison += f"**Provider por defecto:** {AIConfig.DEFAULT_AI_PROVIDER}\n"
        
        return comparison
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas de los providers
        """
        return {
            'total_providers': len(self.providers),
            'available_providers': self.get_available_providers(),
            'default_provider': AIConfig.DEFAULT_AI_PROVIDER,
            'default_available': AIConfig.DEFAULT_AI_PROVIDER in self.providers
        }

# Instancia global del manager
ai_manager = AIProviderManager()

# Funciones de conveniencia para usar en la app principal
def analyze_market(message: str, market_data: Dict[str, Any], provider: str = None) -> str:
    """
    Función principal para análisis de mercado
    Args:
        message: Consulta del usuario
        market_data: Datos del mercado
        provider: Provider específico (opcional)
    """
    if provider:
        return ai_manager.analyze_with_provider(provider, message, market_data)
    else:
        return ai_manager.analyze_with_default(message, market_data)

def get_available_ais() -> List[str]:
    """Retorna lista de IAs disponibles"""
    return ai_manager.get_available_providers()

def get_ai_comparison() -> str:
    """Retorna comparación de IAs disponibles"""
    return ai_manager.get_provider_comparison()

def get_ai_stats() -> Dict[str, Any]:
    """Retorna estadísticas de IAs"""
    return ai_manager.get_provider_stats()