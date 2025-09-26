"""
Gestor centralizado de providers de IA para Trading Assistant
Solo Claude y Gemini - Sin OpenAI
"""
from typing import Dict, List, Optional, Any
import logging
from .base_ai import BaseAI
from .claude_ai import ClaudeAI
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
        """Inicializa solo Claude y Gemini"""
        
        # Claude/Anthropic
        try:
            if AIConfig.ANTHROPIC_API_KEY and AIConfig.ANTHROPIC_API_KEY.startswith('sk-ant-'):
                claude_provider = ClaudeAI()
                if claude_provider.is_available():
                    self.providers['claude'] = claude_provider
                    self.logger.info("✅ Claude (Anthropic) inicializado correctamente")
                else:
                    self.logger.warning("⚠️ Claude no disponible (API key inválida)")
            else:
                self.logger.warning("⚠️ Claude no disponible (falta API key)")
        except Exception as e:
            self.logger.error(f"❌ Error inicializando Claude: {e}")
        
        # Gemini/Google - Con manejo especial para la API
        try:
            if AIConfig.GEMINI_API_KEY and AIConfig.GEMINI_API_KEY.startswith('AIza'):
                gemini_provider = GeminiAI()
                if gemini_provider.is_available():
                    self.providers['gemini'] = gemini_provider
                    self.logger.info("✅ Gemini (Google) inicializado correctamente")
                else:
                    self.logger.warning("⚠️ Gemini no disponible (API key inválida)")
            else:
                self.logger.warning("⚠️ Gemini no disponible (falta API key)")
        except Exception as e:
            self.logger.error(f"❌ Error inicializando Gemini: {e}")
            # Continuar sin Gemini si falla
    
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
            if not available:
                return """❌ **No hay providers de IA disponibles**

**Problema detectado:**
- Gemini falla en la inicialización: `GenerativeModel` no encontrado
- Versión incompatible de google-generativeai

**Solución:**
```bash
# Desinstalar versión actual
pip uninstall google-generativeai

# Instalar versión compatible
pip install google-generativeai==0.4.0
```

**O como alternativa temporal, solo usar Claude:**
```bash
# En tu .env, cambia:
DEFAULT_AI_PROVIDER=claude
```

**Mientras tanto, puedes usar:**
- Datos del mercado en tiempo real
- Gráficos profesionales
- Métricas globales del mercado
            """
            return f"❌ Provider '{provider_name}' no disponible. Providers disponibles: {available}"
        
        self.logger.debug(f"[{provider_name}] Analyzing message: {message}")
  
        try:
            result = provider.analyze_market(message, market_data)
            self.logger.debug(f"[{provider_name}] Response length: {len(result) if result else 0}")
            return result
        except Exception as e:
            self.logger.exception(f"❌ Exception in provider {provider_name}")
            return provider.handle_error(e, "Analysis Error")
    
    def analyze_with_default(self, message: str, market_data: Dict[str, Any]) -> str:
        """
        Analiza usando el provider por defecto
        """
        provider = self.get_default_provider()
        if not provider:
            return """❌ **No hay providers de IA disponibles**
            
**Estado actual:**
- Claude: Configurado correctamente ✅
- Gemini: Error de inicialización ❌

**Solución rápida - Actualizar google-generativeai:**
```bash
pip install --upgrade google-generativeai==0.4.0
```

**O usar solo Claude temporalmente:**
En tu archivo `.env`, cambia:
```bash
DEFAULT_AI_PROVIDER=claude
```

**Características disponibles sin IA:**
- 📊 Gráficos en tiempo real con Binance API
- 💰 Precios actualizados de CoinGecko
- 📈 Métricas globales del mercado
- 😨 Índice Fear & Greed
- 🔧 Análisis técnico visual (MA20, MA50)
            """
        
        return provider.analyze_market(message, market_data)
    
    def get_provider_comparison(self) -> str:
        """
        Genera una comparación de todos los providers disponibles
        """
        if not self.providers:
            return """## ❌ No hay providers de IA disponibles

**Problema principal:** Gemini no se puede inicializar debido a incompatibilidad de versiones.

**Soluciones:**

### 🔧 Opción 1: Actualizar Gemini
```bash
pip install --upgrade google-generativeai==0.4.0
```

### 🔧 Opción 2: Usar solo Claude
En `.env`:
```bash
DEFAULT_AI_PROVIDER=claude
```

### 🔧 Opción 3: Verificar versiones
```bash
pip list | grep -E "(google-generativeai|anthropic)"
```
            """
        
        comparison = "## 🤖 PROVIDERS DE IA DISPONIBLES\n\n"
        
        for name, provider in self.providers.items():
            comparison += f"### {provider.get_provider_name()}\n"
            comparison += f"**Estado:** ✅ Disponible\n"
            comparison += f"**Fortalezas:**\n"
            
            for strength in provider.get_strengths():
                comparison += f"- {strength}\n"
            
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