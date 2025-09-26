"""
Configuración centralizada para Trading Assistant Pro
"""
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class Config:
    """Configuración principal de la aplicación"""
    
    # Configuración de la aplicación
    APP_TITLE = "Trading Assistant Pro"
    APP_ICON = "📈"
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    APP_PORT = int(os.getenv('APP_PORT', 8501))
    
    # APIs de mercado
    COINGECKO_API = "https://api.coingecko.com/api/v3"
    FEAR_GREED_API = "https://api.alternative.me/fng/"
    BINANCE_API = "https://api.binance.com/api/v3"
    
    # Configuración de datos
    DEFAULT_SYMBOLS = {
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD', 
        'Solana': 'SOL-USD',
        'Cardano': 'ADA-USD',
        'Polygon': 'MATIC-USD',
        'Avalanche': 'AVAX-USD'
    }
    
    TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
    DEFAULT_TIMEFRAME = '1h'
    
    # Cache TTL (segundos)
    PRICE_CACHE_TTL = 30
    MARKET_DATA_CACHE_TTL = 60
    CHART_CACHE_TTL = 300

class AIConfig:
    """Configuración específica para providers de IA"""
    
    # OpenAI
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')
    OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', 1000))
    OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', 0.7))
    
    # Anthropic Claude
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-3-sonnet-20240229')
    CLAUDE_MAX_TOKENS = int(os.getenv('CLAUDE_MAX_TOKENS', 1000))
    
    # Google Gemini
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-pro')
    
    # Configuración general de IA
    DEFAULT_AI_PROVIDER = os.getenv('DEFAULT_AI_PROVIDER', 'gemini')
    AI_REQUEST_TIMEOUT = int(os.getenv('AI_REQUEST_TIMEOUT', 30))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))

class TradingConfig:
    """Configuración específica para trading"""
    
    # Risk Management por defecto
    DEFAULT_RISK_PERCENTAGE = 2.0  # 2% del capital por trade
    DEFAULT_STOP_LOSS = {
        'BTC': 5.0,   # 5% stop loss para Bitcoin
        'ETH': 8.0,   # 8% stop loss para Ethereum  
        'SOL': 12.0,  # 12% stop loss para Solana (mayor volatilidad)
        'ALT': 15.0   # 15% stop loss para altcoins
    }
    
    # Indicadores técnicos
    RSI_PERIOD = 14
    MA_SHORT_PERIOD = 20
    MA_LONG_PERIOD = 50
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2
    
    # Niveles de Fear & Greed
    FEAR_GREED_LEVELS = {
        'extreme_fear': (0, 25),
        'fear': (25, 45), 
        'neutral': (45, 55),
        'greed': (55, 75),
        'extreme_greed': (75, 100)
    }

# Funciones de validación corregidas
def is_api_key_valid(api_key, provider_type):
    """Valida si una API key tiene formato correcto"""
    if not api_key:
        return False
        
    # Verificar que no sea placeholder
    placeholder_keys = [
        'sk-your-openai-key-here',
        'sk-ant-your-claude-key-here', 
        'your-gemini-key-here',
        'your-google-key-here'
    ]
    
    if api_key in placeholder_keys:
        return False
    
    # Verificar formato básico
    if provider_type == 'openai':
        return api_key.startswith('sk-') and len(api_key) > 20
    elif provider_type == 'anthropic':
        return api_key.startswith('sk-ant-') and len(api_key) > 30
    elif provider_type == 'gemini':
        # Las keys de Gemini empiezan con "AIza" y son largas
        return api_key.startswith('AIza') and len(api_key) > 35
    
    return True

def get_available_ai_providers():
    """Retorna lista de providers de IA disponibles según las API keys - VERSIÓN CORREGIDA"""
    providers = []
    
    # Debug: imprimir las keys para verificar
    if Config.DEBUG:
        print("=== DEBUG API KEYS ===")
        print(f"OPENAI_API_KEY: {AIConfig.OPENAI_API_KEY[:10] if AIConfig.OPENAI_API_KEY else 'None'}...")
        print(f"ANTHROPIC_API_KEY: {AIConfig.ANTHROPIC_API_KEY[:15] if AIConfig.ANTHROPIC_API_KEY else 'None'}...")
        print(f"GEMINI_API_KEY: {AIConfig.GEMINI_API_KEY[:10] if AIConfig.GEMINI_API_KEY else 'None'}...")
    
    # Verificar OpenAI
    if is_api_key_valid(AIConfig.OPENAI_API_KEY, 'openai'):
        providers.append('openai')
        if Config.DEBUG:
            print("✅ OpenAI añadido")
    elif Config.DEBUG:
        print("❌ OpenAI no válido")
    
    # Verificar Anthropic/Claude
    if is_api_key_valid(AIConfig.ANTHROPIC_API_KEY, 'anthropic'):
        providers.append('claude')
        if Config.DEBUG:
            print("✅ Claude añadido")
    elif Config.DEBUG:
        print("❌ Claude no válido")
    
    # Verificar Gemini - CORRECCIÓN PRINCIPAL
    if is_api_key_valid(AIConfig.GEMINI_API_KEY, 'gemini'):
        providers.append('gemini')
        if Config.DEBUG:
            print("✅ Gemini añadido")
    elif Config.DEBUG:
        print(f"❌ Gemini no válido - Key: {AIConfig.GEMINI_API_KEY[:15] if AIConfig.GEMINI_API_KEY else 'None'}...")
        
    if Config.DEBUG:
        print(f"Providers finales: {providers}")
        print("====================")
        
    return providers

def validate_config():
    """Valida que la configuración esté correcta"""
    errors = []
    
    # Obtener providers disponibles usando la función corregida
    available_providers = get_available_ai_providers()
    
    if not available_providers:
        errors.append("⚠️ No se encontró ninguna API key de IA válida. Agrega al menos una en el archivo .env")
    
    # Validar provider por defecto
    valid_providers = ['openai', 'claude', 'gemini']
    if AIConfig.DEFAULT_AI_PROVIDER not in valid_providers:
        errors.append(f"❌ Provider de IA inválido: {AIConfig.DEFAULT_AI_PROVIDER}. Debe ser uno de: {valid_providers}")
    
    # Validar que el provider por defecto esté disponible
    if AIConfig.DEFAULT_AI_PROVIDER not in available_providers and available_providers:
        errors.append(f"⚠️ Provider por defecto '{AIConfig.DEFAULT_AI_PROVIDER}' no está disponible. Providers disponibles: {available_providers}")
    
    return errors

def get_symbol_display_name(symbol):
    """Convierte símbolo técnico a nombre legible"""
    symbol_names = {
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum',
        'SOL-USD': 'Solana', 
        'ADA-USD': 'Cardano',
        'MATIC-USD': 'Polygon',
        'AVAX-USD': 'Avalanche'
    }
    return symbol_names.get(symbol, symbol)

def get_provider_status():
    """Retorna estado detallado de cada provider"""
    status = {}
    
    # OpenAI
    status['openai'] = {
        'available': is_api_key_valid(AIConfig.OPENAI_API_KEY, 'openai'),
        'model': AIConfig.OPENAI_MODEL,
        'name': 'GPT-4 (OpenAI)'
    }
    
    # Anthropic Claude  
    status['claude'] = {
        'available': is_api_key_valid(AIConfig.ANTHROPIC_API_KEY, 'anthropic'),
        'model': AIConfig.CLAUDE_MODEL,
        'name': 'Claude (Anthropic)'
    }
    
    # Google Gemini
    status['gemini'] = {
        'available': is_api_key_valid(AIConfig.GEMINI_API_KEY, 'gemini'),
        'model': AIConfig.GEMINI_MODEL,
        'name': 'Gemini (Google)'
    }
    
    return status

def debug_ai_config():
    """Función de debug para verificar configuración de AI"""
    print("=== DEBUG DETALLADO ===")
    print(f"OPENAI_API_KEY: '{AIConfig.OPENAI_API_KEY}'")
    print(f"ANTHROPIC_API_KEY: '{AIConfig.ANTHROPIC_API_KEY}'")
    print(f"GEMINI_API_KEY: '{AIConfig.GEMINI_API_KEY}'")
    print(f"DEFAULT_AI_PROVIDER: {AIConfig.DEFAULT_AI_PROVIDER}")
    
    print("\n--- VALIDACIONES INDIVIDUALES ---")
    if AIConfig.OPENAI_API_KEY:
        print(f"OpenAI válida: {is_api_key_valid(AIConfig.OPENAI_API_KEY, 'openai')}")
    if AIConfig.ANTHROPIC_API_KEY:
        print(f"Claude válida: {is_api_key_valid(AIConfig.ANTHROPIC_API_KEY, 'anthropic')}")
    if AIConfig.GEMINI_API_KEY:
        print(f"Gemini válida: {is_api_key_valid(AIConfig.GEMINI_API_KEY, 'gemini')}")
        print(f"Gemini starts with AIza: {AIConfig.GEMINI_API_KEY.startswith('AIza') if AIConfig.GEMINI_API_KEY else False}")
        print(f"Gemini length: {len(AIConfig.GEMINI_API_KEY) if AIConfig.GEMINI_API_KEY else 0}")
    
    print(f"\nProviders disponibles: {get_available_ai_providers()}")
    print("========================")

if __name__ == "__main__":
    # Test de configuración
    print("🔍 INICIANDO DIAGNÓSTICO DE CONFIGURACIÓN...")
    
    # Debug detallado si está habilitado
    if Config.DEBUG:
        debug_ai_config()
    
    # Validación normal
    errors = validate_config()
    if errors:
        print("❌ Errores de configuración:")
        for error in errors:
            print(f"  {error}")
    else:
        print("✅ Configuración válida")
        
    available = get_available_ai_providers()
    print(f"📊 Providers de IA disponibles: {available}")
    print(f"🤖 Provider por defecto: {AIConfig.DEFAULT_AI_PROVIDER}")
    
    # Mostrar estado de providers
    provider_status = get_provider_status()
    print("\n🔍 Estado de providers:")
    for name, status in provider_status.items():
        status_icon = "✅" if status['available'] else "❌"
        print(f"  {status_icon} {status['name']}: {'Disponible' if status['available'] else 'No configurado'}")
    
    print(f"\n🎯 Recomendación: Usa el provider '{available[0] if available else 'ninguno'}'")