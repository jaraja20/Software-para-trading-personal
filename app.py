"""
Trading Assistant Pro - Aplicación Principal
Análisis avanzado con múltiples providers de IA para trading de criptomonedas
VERSIÓN COMPATIBLE CON PYTHON 3.8
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import numpy as np
import logging
import sys
from typing import Dict, List, Optional, Any

# Importar módulos propios con manejo de errores (ANTES de cualquier comando de Streamlit)
try:
    from config import Config, AIConfig, TradingConfig, validate_config
    from ai_providers import analyze_market, get_available_ais, get_ai_comparison, get_ai_stats
    config_loaded = True
except ImportError as e:
    config_loaded = False
    import_error = str(e)
except Exception as e:
    config_loaded = False
    import_error = str(e)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la página (DEBE SER LO PRIMERO)
if config_loaded:
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
else:
    st.set_page_config(
        page_title="Trading Assistant Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Mostrar advertencias DESPUÉS de set_page_config
if sys.version_info < (3, 9):
    st.warning("⚠️ Estás usando Python 3.8. Para mejor compatibilidad, considera actualizar a Python 3.10+")

# Mostrar errores de importación si los hay
if not config_loaded:
    st.error(f"❌ Error importando configuración: {import_error}")

# CSS personalizado para tema dark y estilo profesional
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
    }
    .main-header {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
    }
    .ai-provider-card {
        background: #1e293b;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #1e293b;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #334155;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .price-positive {
        color: #10b981 !important;
    }
    .price-negative {
        color: #ef4444 !important;
    }
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 1rem;
        border-radius: 10px;
    }
    .stSelectbox > div > div {
        background-color: #1e293b;
        color: #f8fafc;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        word-wrap: break-word;
    }
    .user-message {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        margin-left: 15%;
        border-bottom-right-radius: 4px;
    }
    .ai-message {
        background: #374151;
        color: #f3f4f6;
        margin-right: 15%;
        border-bottom-left-radius: 4px;
        border-left: 3px solid #3b82f6;
    }
    .sidebar-section {
        background: #1e293b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #334155;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    .status-online { background-color: #10b981; }
    .status-offline { background-color: #ef4444; }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Scrollbar personalizado */
    .element-container::-webkit-scrollbar {
        width: 6px;
    }
    .element-container::-webkit-scrollbar-track {
        background: #1e293b;
    }
    .element-container::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Inicializa el estado de la sesión"""
    defaults = {
        'chat_history': [
            {
                "role": "ai", 
                "message": "👋 **¡Bienvenido a Trading Assistant Pro!**\n\nSoy tu asistente especializado en criptomonedas con acceso a múltiples IAs:\n\n• **Claude** - Análisis profundo y contextual\n• **GPT-4** - Versatilidad y creatividad\n• **Gemini** - Análisis multimodal rápido\n\n¿Qué te gustaría analizar hoy?",
                "provider": "system"
            }
        ],
        'current_symbol': 'BTC-USD',
        'current_timeframe': '1h',
        'selected_ai': None,
        'show_ai_comparison': False,
        'chart_data_cache': {},
        'last_refresh': datetime.now()
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Configurar IA por defecto
    if st.session_state.selected_ai is None and config_loaded:
        try:
            available_ais = get_available_ais()
            if available_ais:
                default_ai = AIConfig.DEFAULT_AI_PROVIDER if AIConfig.DEFAULT_AI_PROVIDER in available_ais else available_ais[0]
                st.session_state.selected_ai = default_ai
        except:
            st.session_state.selected_ai = None

def validate_and_show_config():
    """Valida configuración y muestra advertencias si es necesario"""
    if not config_loaded:
        st.error("❌ **Error de configuración:** No se pudieron cargar los módulos necesarios")
        with st.expander("💡 **Solución:**"):
            st.markdown("""
            **Para Python 3.8, instala versiones específicas compatibles:**
            
            ```bash
            pip install streamlit==1.25.0
            pip install plotly==5.15.0
            pip install pandas==2.0.3
            pip install numpy==1.24.4
            pip install requests==2.31.0
            pip install python-dotenv==1.0.0
            pip install openai==0.28.1
            pip install anthropic==0.3.11
            pip install google-generativeai==0.3.1
            ```
            
            **O actualiza Python a 3.10+ (recomendado):**
            - Descarga desde https://python.org
            - Reinstala las librerías con versiones más recientes
            """)
        return False
    
    try:
        config_errors = validate_config()
        if config_errors:
            st.error("⚠️ **Configuración incompleta:**")
            for error in config_errors:
                st.error(f"• {error}")
            
            with st.expander("💡 **Cómo configurar APIs**"):
                st.markdown("""
                Para habilitar funcionalidad completa de IA, agrega al menos una API key en tu archivo `.env`:
                
                ```bash
                # OpenAI (GPT-4) - Versátil y balanceado
                OPENAI_API_KEY=sk-your-openai-key-here
                
                # Anthropic (Claude) - Análisis profundo  
                ANTHROPIC_API_KEY=sk-ant-your-claude-key-here
                
                # Google (Gemini) - Rápido y económico
                GEMINI_API_KEY=your-gemini-key-here
                ```
                
                **Dónde obtener las keys:**
                - OpenAI: https://platform.openai.com/api-keys
                - Claude: https://console.anthropic.com/
                - Gemini: https://ai.google.dev/
                """)
            
            return False
        return True
    except Exception as e:
        st.error(f"❌ Error validando configuración: {e}")
        return False

# Funciones de datos del mercado con manejo mejorado de errores
@st.cache_data(ttl=30, show_spinner=False)
def get_crypto_prices():
    """Obtiene precios actuales de las principales cryptos"""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': 'bitcoin,ethereum,solana,cardano,polygon-ecosystem-token,avalanche-2',
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_24hr_vol': 'true',
            'include_market_cap': 'true'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error obteniendo precios: {e}")
        return None

@st.cache_data(ttl=300, show_spinner=False)
def get_fear_greed_index():
    """Obtiene el índice de miedo y codicia"""
    try:
        response = requests.get("https://api.alternative.me/fng/", timeout=10)
        response.raise_for_status()
        return response.json()['data'][0]
    except Exception as e:
        logger.error(f"Error obteniendo Fear & Greed: {e}")
        return None

@st.cache_data(ttl=300, show_spinner=False)
def get_market_data():
    """Obtiene datos generales del mercado"""
    try:
        response = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        response.raise_for_status()
        return response.json()['data']
    except Exception as e:
        logger.error(f"Error obteniendo datos del mercado: {e}")
        return None

@st.cache_data(ttl=60, show_spinner=False)
def get_candlestick_data(symbol='BTC-USD', timeframe='1h', days=30):
    """Obtiene datos de velas para el gráfico usando Binance API"""
    try:
        # Mapeo de símbolos
        symbol_map = {
            'BTC-USD': 'BTCUSDT',
            'ETH-USD': 'ETHUSDT',
            'SOL-USD': 'SOLUSDT',
            'ADA-USD': 'ADAUSDT',
            'MATIC-USD': 'MATICUSDT',
            'AVAX-USD': 'AVAXUSDT'
        }
        binance_symbol = symbol_map.get(symbol, 'BTCUSDT')

        # Mapear timeframe
        interval_map = {
            '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h',
            '4h': '4h', '1d': '1d', '1w': '1w', '1M': '1M'
        }
        interval = interval_map.get(timeframe, '1h')

        # Llamada a la API de Binance
        url = "https://api.binance.com/api/v3/klines"
        limit = min(days * 24, 1000)
        params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Convertir a DataFrame
        df = pd.DataFrame(data, columns=[
            "OpenTime", "Open", "High", "Low", "Close", "Volume",
            "CloseTime", "QuoteAssetVolume", "Trades",
            "TakerBuyBase", "TakerBuyQuote", "Ignore"
        ])
        df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms")
        df.set_index("OpenTime", inplace=True)
        
        # Convertir a float de manera compatible con Python 3.8
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        logger.error(f"Error obteniendo datos de Binance para {symbol}: {e}")
        # Generar datos de demostración en caso de error
        return generate_demo_data(symbol, timeframe, days)

def generate_demo_data(symbol, timeframe, days):
    """Genera datos de demostración en caso de que falle la API"""
    try:
        # Precios base por símbolo
        base_prices = {
            'BTC-USD': 43000, 'ETH-USD': 2600, 'SOL-USD': 98,
            'ADA-USD': 0.52, 'MATIC-USD': 0.85, 'AVAX-USD': 37
        }
        
        base_price = base_prices.get(symbol, 1000)
        
        # Generar fechas
        if timeframe in ['5m', '15m', '30m']:
            freq_minutes = int(timeframe.replace('m', ''))
            periods = min(days * 24 * (60 // freq_minutes), 500)
            freq = f'{freq_minutes}min'
        elif timeframe == '1h':
            freq = 'H'
            periods = days * 24
        elif timeframe == '4h':
            freq = '4H' 
            periods = days * 6
        elif timeframe == '1d':
            freq = 'D'
            periods = days
        else:
            freq = 'W'
            periods = days // 7
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days), 
            periods=periods, 
            freq=freq
        )
        
        # Generar precios con walk aleatorio
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = [base_price]
        
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            new_price = max(new_price, prices[-1] * 0.95)  # Limitar pérdidas
            prices.append(new_price)
        
        prices = prices[1:]
        
        # Generar OHLC
        ohlc_data = []
        for i, price in enumerate(prices):
            if i == 0:
                open_price = base_price
            else:
                open_price = ohlc_data[-1]['Close']
                
            close_price = price
            high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.03)
            low_price = min(open_price, close_price) * np.random.uniform(0.97, 1.0)
            volume = np.random.uniform(1000000, 10000000)
                
            ohlc_data.append({
                'Open': open_price,
                'High': high_price, 
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
        
        return pd.DataFrame(ohlc_data, index=dates)
    
    except Exception as e:
        logger.error(f"Error generando datos demo: {e}")
        return pd.DataFrame()

def create_candlestick_chart(data, symbol, timeframe):
    """Crea gráfico de velas profesional con Plotly"""
    if data.empty:
        return None
    
    try:
        # Crear subplots: precio (80%) y volumen (20%)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} - {timeframe.upper()}', 'Volumen'),
            row_heights=[0.8, 0.2]
        )
        
        # Gráfico de velas principal
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Precio",
                increasing_line_color='#10b981',
                decreasing_line_color='#ef4444',
                increasing_fillcolor='rgba(16, 185, 129, 0.3)',
                decreasing_fillcolor='rgba(239, 68, 68, 0.3)'
            ),
            row=1, col=1
        )
        
        # Medias móviles si hay suficientes datos
        if len(data) >= 20:
            data_copy = data.copy()
            data_copy['MA20'] = data_copy['Close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=data_copy.index, 
                    y=data_copy['MA20'],
                    name='MA20', 
                    line=dict(color='#3b82f6', width=2),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        if len(data) >= 50:
            data_copy = data.copy()
            data_copy['MA50'] = data_copy['Close'].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(
                    x=data_copy.index, 
                    y=data_copy['MA50'],
                    name='MA50', 
                    line=dict(color='#f59e0b', width=2),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        # Gráfico de volumen con colores
        colors = ['#10b981' if data['Close'].iloc[i] >= data['Open'].iloc[i] 
                  else '#ef4444' for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name="Volumen",
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Layout profesional
        fig.update_layout(
            title={
                'text': f"<b>{symbol}</b> - Análisis Técnico [{timeframe.upper()}]",
                'x': 0.5,
                'font': {'size': 20, 'color': '#f8fafc'}
            },
            template="plotly_dark",
            height=650,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left", 
                x=0.01,
                bgcolor="rgba(30, 41, 59, 0.8)"
            ),
            xaxis_rangeslider_visible=False,
            plot_bgcolor='#0f172a',
            paper_bgcolor='#0f172a',
            font=dict(color='#f8fafc'),
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Personalizar ejes
        fig.update_yaxes(title_text="Precio (USD)", row=1, col=1, gridcolor='#334155')
        fig.update_yaxes(title_text="Volumen", row=2, col=1, gridcolor='#334155')
        fig.update_xaxes(gridcolor='#334155')
        
        # Añadir anotación de cambio de precio
        if len(data) > 1:
            price_change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            color = '#10b981' if price_change > 0 else '#ef4444'
            
            fig.add_annotation(
                x=data.index[-1], 
                y=data['Close'].iloc[-1],
                text=f"{price_change:+.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor=color,
                font=dict(color=color, size=12),
                bgcolor="rgba(15, 23, 42, 0.8)",
                bordercolor=color
            )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creando gráfico: {e}")
        return None

def create_market_summary(prices_data, fear_greed_data, market_data):
    """Crea resumen del mercado para contexto de IA"""
    summary = {
        'prices': prices_data,
        'fear_greed': fear_greed_data,
        'global_data': market_data,
        'timestamp': datetime.now().isoformat()
    }
    return summary

def display_sidebar():
    """Renderiza la sidebar con controles y configuración"""
    st.sidebar.title("🎛️ Panel de Control")
    
    # Información de estado
    try:
        if config_loaded:
            ai_stats = get_ai_stats()
            status_color = "status-online" if ai_stats['total_providers'] > 0 else "status-offline"
            status_text = f"{ai_stats['total_providers']} IA{'s' if ai_stats['total_providers'] != 1 else ''} disponible{'s' if ai_stats['total_providers'] != 1 else ''}"
        else:
            status_color = "status-offline"
            status_text = "Config. no cargada"
        
        st.sidebar.markdown(f"""
        <div class="sidebar-section">
            <h4>📡 Estado del Sistema</h4>
            <p><span class="status-indicator {status_color}"></span>{status_text}</p>
            <small>Última actualización: {datetime.now().strftime('%H:%M:%S')}</small>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"Error en estado: {e}")
    
    # Selector de IA (solo si la configuración está cargada)
    if config_loaded:
        try:
            available_ais = get_available_ais()
            if available_ais:
                st.sidebar.markdown("### 🤖 Configuración de IA")
                
                # Mapeo de nombres técnicos a nombres amigables
                ai_display_names = {
                    'claude': '🧠 Claude (Análisis profundo)',
                    'openai': '💡 GPT-4 (Versatilidad)', 
                    'gemini': '⚡ Gemini (Velocidad)'
                }
                
                display_options = [ai_display_names.get(ai, ai) for ai in available_ais]
                
                selected_display = st.sidebar.selectbox(
                    "Seleccionar IA:",
                    display_options,
                    index=available_ais.index(st.session_state.selected_ai) if st.session_state.selected_ai in available_ais else 0,
                    help="Cada IA tiene fortalezas específicas para diferentes tipos de análisis"
                )
                
                # Mapear de vuelta al nombre técnico
                reverse_mapping = {v: k for k, v in ai_display_names.items()}
                st.session_state.selected_ai = reverse_mapping.get(selected_display, available_ais[0])
                
                # Botón de comparación de IAs
                if st.sidebar.button("📊 Comparar IAs", use_container_width=True):
                    st.session_state.show_ai_comparison = not st.session_state.show_ai_comparison
        except Exception as e:
            st.sidebar.error(f"Error configurando IA: {e}")
    
    # Configuración de gráfico
    st.sidebar.markdown("### 📊 Configuración de Gráfico")
    
    # Selector de criptomoneda
    default_symbols = {
        'Bitcoin (BTC)': 'BTC-USD',
        'Ethereum (ETH)': 'ETH-USD', 
        'Solana (SOL)': 'SOL-USD',
        'Cardano (ADA)': 'ADA-USD',
        'Polygon (MATIC)': 'MATIC-USD',
        'Avalanche (AVAX)': 'AVAX-USD'
    }
    
    selected_crypto = st.sidebar.selectbox(
        "Criptomoneda:",
        list(default_symbols.keys()),
        index=list(default_symbols.values()).index(st.session_state.current_symbol) if st.session_state.current_symbol in default_symbols.values() else 0
    )
    
    st.session_state.current_symbol = default_symbols[selected_crypto]
    
    # Selector de timeframe
    timeframe_display = {
        '5m': '5 Minutos', '15m': '15 Minutos', '30m': '30 Minutos',
        '1h': '1 Hora', '4h': '4 Horas', '1d': '1 Día', 
        '1w': '1 Semana', '1M': '1 Mes'
    }
    
    selected_timeframe_display = st.sidebar.selectbox(
        "Timeframe:",
        list(timeframe_display.values()),
        index=list(timeframe_display.keys()).index(st.session_state.current_timeframe) if st.session_state.current_timeframe in timeframe_display else 3
    )
    
    # Mapear de vuelta al valor técnico
    reverse_timeframe = {v: k for k, v in timeframe_display.items()}
    st.session_state.current_timeframe = reverse_timeframe[selected_timeframe_display]
    
    # Controles adicionales
    st.sidebar.markdown("### 🔧 Controles")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Limpiar Chat", use_container_width=True):
            st.session_state.chat_history = [
                {"role": "ai", "message": "👋 Chat reiniciado. ¿En qué puedo ayudarte?", "provider": "system"}
            ]
            st.rerun()

def display_ai_comparison():
    """Muestra comparación detallada de IAs disponibles"""
    if st.session_state.show_ai_comparison and config_loaded:
        st.markdown("## 🤖 Comparación de Providers de IA")
        
        try:
            comparison_text = get_ai_comparison()
            st.markdown(comparison_text)
            
            # Tabla de comparación rápida
            st.markdown("### ⚡ Comparación Rápida")
            
            comparison_data = {
                'Provider': ['Claude', 'GPT-4', 'Gemini'],
                'Fortaleza Principal': [
                    'Análisis profundo y contextual',
                    'Versatilidad y creatividad', 
                    'Velocidad y análisis multimodal'
                ],
                'Mejor para': [
                    'Análisis de riesgo complejo',
                    'Estrategias creativas',
                    'Análisis rápido de tendencias'
                ],
                'Costo': ['Medio', 'Alto', 'Bajo'],
                'Velocidad': ['Media', 'Media', 'Alta']
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
        except Exception as e:
            st.error(f"Error mostrando comparación: {e}")

def main():
    """Función principal de la aplicación"""
    
    # Inicializar estado
    initialize_session_state()
    
    # Validar configuración
    config_valid = validate_and_show_config()
    
    # Header principal
    app_title = Config.APP_TITLE if config_loaded else "Trading Assistant Pro"
    app_icon = Config.APP_ICON if config_loaded else "📈"
    
    try:
        ai_count = len(get_available_ais()) if config_loaded else 0
    except:
        ai_count = 0
    
    st.markdown(f"""
    <div class="main-header">
        <h1>{app_icon} {app_title}</h1>
        <small>🤖 {ai_count} IA{'s' if ai_count != 1 else ''} disponible{'s' if ai_count != 1 else ''}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    display_sidebar()
    
    # Comparación de IAs (si está activada)
    display_ai_comparison()
    
    # Obtener datos del mercado
    with st.spinner("📡 Cargando datos del mercado..."):
        prices_data = get_crypto_prices()
        market_data = get_market_data() 
        fear_greed_data = get_fear_greed_index()
    
    # Mostrar métricas principales
    if prices_data:
        st.markdown("### 💰 Precios en Tiempo Real")
        
        # Preparar datos para mostrar
        metrics_data = []
        for crypto_id, display_name in [
            ('bitcoin', 'Bitcoin'), ('ethereum', 'Ethereum'), 
            ('solana', 'Solana'), ('cardano', 'Cardano')
        ]:
            if crypto_id in prices_data:
                data = prices_data[crypto_id]
                metrics_data.append({
                    'name': display_name,
                    'price': data['usd'],
                    'change': data['usd_24h_change'],
                    'volume': data.get('usd_24h_vol', 0),
                    'market_cap': data.get('market_cap', 0)
                })
        
        # Mostrar métricas en columnas
        if metrics_data:
            cols = st.columns(len(metrics_data))
            for i, metric in enumerate(metrics_data):
                with cols[i]:
                    # Seleccionar emoji según la crypto
                    emoji = "₿" if metric['name'] == 'Bitcoin' else "⟠" if metric['name'] == 'Ethereum' else "◎" if metric['name'] == 'Solana' else "🔷"
                    price_display = f"${metric['price']:,.2f}" if metric['price'] >= 1 else f"${metric['price']:.4f}"
                    
                    st.metric(
                        f"{emoji} {metric['name']}", 
                        price_display,
                        f"{metric['change']:+.2f}%"
                    )
    else:
        st.warning("⚠️ No se pudieron cargar los precios en tiempo real. Usando datos de demostración.")
    
    # Layout principal: Gráfico y Chat
    col_chart, col_chat = st.columns([2.2, 1])
    
    with col_chart:
        st.markdown(f"### 📊 Gráfico de {st.session_state.current_symbol.replace('-USD', '')} - {st.session_state.current_timeframe.upper()}")
        
        # Obtener datos del gráfico
        with st.spinner("📈 Cargando datos del gráfico..."):
            chart_data = get_candlestick_data(
                st.session_state.current_symbol, 
                st.session_state.current_timeframe, 
                days=30
            )
        
        if not chart_data.empty:
            fig = create_candlestick_chart(
                chart_data, 
                st.session_state.current_symbol.replace('-USD', ''),
                st.session_state.current_timeframe
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar estadísticas del gráfico
                try:
                    current_price = chart_data['Close'].iloc[-1]
                    price_change = ((current_price / chart_data['Close'].iloc[0]) - 1) * 100
                    volume_avg = chart_data['Volume'].mean()
                    
                    st.markdown(f"""
                    **📊 Estadísticas del Gráfico:**
                    - **Precio actual:** ${current_price:,.2f}
                    - **Cambio del período:** {price_change:+.2f}%
                    - **Volumen promedio:** {volume_avg:,.0f}
                    - **Datos desde:** {chart_data.index[0].strftime('%Y-%m-%d %H:%M')}
                    """)
                except Exception as e:
                    st.error(f"Error mostrando estadísticas: {e}")
            else:
                st.error("❌ No se pudo generar el gráfico")
        else:
            st.error("❌ No se pudieron cargar los datos del gráfico")
    
    with col_chat:
        st.markdown("### 🤖 Chat con IA Especializada")
        
        # Mostrar información de la IA actual
        if st.session_state.selected_ai and config_loaded:
            ai_names = {
                'claude': '🧠 Claude (Anthropic)',
                'openai': '💡 GPT-4 (OpenAI)', 
                'gemini': '⚡ Gemini (Google)'
            }
            current_ai_name = ai_names.get(st.session_state.selected_ai, st.session_state.selected_ai)
            
            st.markdown(f"""
            <div class="ai-provider-card">
                <strong>IA Activa:</strong> {current_ai_name}<br>
                <small>💬 {len(st.session_state.chat_history)} mensajes en la sesión</small>
            </div>
            """, unsafe_allow_html=True)
        
        # CONTENEDOR DE CHAT CON ALTURA FIJA Y SCROLL
        st.markdown("""
        <style>
        .chat-container {
            height: 400px;
            overflow-y: auto;
            padding: 1rem;
            background-color: #1e293b;
            border-radius: 8px;
            border: 1px solid #334155;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Crear contenedor de chat con altura fija
        chat_html = '<div class="chat-container">'
        
        # Generar HTML para cada mensaje
        for i, chat in enumerate(st.session_state.chat_history):
            if chat["role"] == "ai":
                provider_info = ""
                if chat.get("provider") != "system":
                    provider_name = chat.get("provider", st.session_state.selected_ai or "IA")
                    ai_names = {
                        'claude': '🧠 Claude',
                        'openai': '💡 GPT-4',
                        'gemini': '⚡ Gemini'
                    }
                    provider_display = ai_names.get(provider_name, provider_name)
                    provider_info = f"<small><em>{provider_display}</em></small><br>"
                
                chat_html += f"""
                <div class="chat-message ai-message">
                    {provider_info}{chat["message"]}
                </div>
                """
            else:
                chat_html += f"""
                <div class="chat-message user-message">
                    {chat["message"]}
                </div>
                """
        
        chat_html += '</div>'
        
        # Mostrar el contenedor de chat
        st.markdown(chat_html, unsafe_allow_html=True)
        
        # Input del chat
        st.markdown("---")
        
        # Ejemplos de preguntas
        with st.expander("💡 Ejemplos de consultas"):
            example_queries = [
                "Analiza Bitcoin en timeframe 4h",
                "Puntos de entrada para Ethereum",
                "Gestión de riesgo para SOL", 
                "Análisis del mercado crypto general",
                "Niveles de soporte y resistencia",
                "Indicadores técnicos actuales"
            ]
            
            for example in example_queries:
                if st.button(f"📝 {example}", key=f"example_{example[:10]}", use_container_width=True):
                    # Procesar directamente la consulta
                    if st.session_state.selected_ai and config_loaded:
                        # Agregar mensaje del usuario
                        st.session_state.chat_history.append({
                            "role": "user", 
                            "message": example
                        })
                        
                        # Preparar contexto del mercado
                        market_summary = create_market_summary(prices_data, fear_greed_data, market_data)
                        
                        # Generar respuesta de IA
                        try:
                            ai_response = analyze_market(
                                example, 
                                market_summary, 
                                provider=st.session_state.selected_ai
                            )
                            
                            st.session_state.chat_history.append({
                                "role": "ai",
                                "message": ai_response,
                                "provider": st.session_state.selected_ai
                            })
                            
                        except Exception as e:
                            error_msg = f"❌ Error procesando consulta: {str(e)}"
                            st.session_state.chat_history.append({
                                "role": "ai",
                                "message": error_msg,
                                "provider": "system"
                            })
                    else:
                        st.error("❌ No hay IAs disponibles")
                    
                    st.rerun()
        
        # Campo de entrada de texto
        user_input = st.text_area(
            "Tu consulta:",
            height=100,
            placeholder="Ej: Analiza Bitcoin, dame puntos de entrada para ETH, qué opinas del mercado...",
            help="Pregunta sobre análisis técnico, tendencias, gestión de riesgo o estrategias de trading",
            key="chat_input_field"
        )
        
        # Botones de acción del chat
        col_send, col_clear_msg = st.columns([2, 1])
        
        with col_send:
            if st.button("📤 Analizar", use_container_width=True, key="send_button") and user_input.strip():
                if not st.session_state.selected_ai or not config_loaded:
                    st.error("❌ No hay IAs disponibles. Configura al menos una API key.")
                else:
                    # Agregar mensaje del usuario
                    st.session_state.chat_history.append({
                        "role": "user", 
                        "message": user_input.strip()
                    })
                    
                    # Preparar contexto del mercado
                    market_summary = create_market_summary(prices_data, fear_greed_data, market_data)
                    
                    # Generar respuesta de IA
                    with st.spinner(f"🤔 {st.session_state.selected_ai.title()} está analizando..."):
                        try:
                            ai_response = analyze_market(
                                user_input.strip(), 
                                market_summary, 
                                provider=st.session_state.selected_ai
                            )
                            
                            st.session_state.chat_history.append({
                                "role": "ai",
                                "message": ai_response,
                                "provider": st.session_state.selected_ai
                            })
                            
                        except Exception as e:
                            error_msg = f"❌ Error procesando consulta: {str(e)}"
                            st.session_state.chat_history.append({
                                "role": "ai",
                                "message": error_msg,
                                "provider": "system"
                            })
                    
                    st.rerun()
        
        with col_clear_msg:
            if st.button("🗑️", use_container_width=True, help="Limpiar último mensaje"):
                if len(st.session_state.chat_history) > 1:
                    st.session_state.chat_history.pop()
                    st.rerun()
    
    # Métricas adicionales del mercado
    if market_data and fear_greed_data:
        st.markdown("---")
        st.markdown("### 📈 Métricas Globales del Mercado")
        
        # Crear métricas en columnas
        try:
            metric_cols = st.columns(6)
            
            with metric_cols[0]:
                market_cap = market_data['total_market_cap']['usd']
                st.metric(
                    "💰 Market Cap Total",
                    f"${market_cap/1e12:.2f}T",
                    help="Capitalización total del mercado crypto"
                )
            
            with metric_cols[1]:
                volume = market_data['total_volume']['usd']
                st.metric(
                    "📊 Volumen 24h",
                    f"${volume/1e9:.1f}B",
                    help="Volumen de trading en 24 horas"
                )
            
            with metric_cols[2]:
                btc_dominance = market_data['market_cap_percentage']['btc']
                st.metric(
                    "₿ Dominancia BTC",
                    f"{btc_dominance:.1f}%",
                    help="Porcentaje del market cap total que representa Bitcoin"
                )
            
            with metric_cols[3]:
                eth_dominance = market_data['market_cap_percentage'].get('eth', 0)
                st.metric(
                    "⟠ Dominancia ETH",  
                    f"{eth_dominance:.1f}%",
                    help="Porcentaje del market cap total que representa Ethereum"
                )
            
            with metric_cols[4]:
                active_cryptos = market_data['active_cryptocurrencies']
                st.metric(
                    "🪙 Cryptos Activas",
                    f"{active_cryptos:,}",
                    help="Número total de criptomonedas activas"
                )
            
            with metric_cols[5]:
                fear_value = int(fear_greed_data['value'])
                fear_classification = fear_greed_data['value_classification']
                
                # Determinar color del delta basado en el valor
                if fear_value <= 25:
                    delta_color = "inverse"  # Extremo miedo
                elif fear_value >= 75:
                    delta_color = "normal"  # Extrema codicia
                else:
                    delta_color = "off"  # Neutral
                
                st.metric(
                    "😨 Fear & Greed",
                    f"{fear_value}/100",
                    fear_classification,
                    delta_color=delta_color,
                    help="Índice de sentimiento del mercado: 0 = Extremo Miedo, 100 = Extrema Codicia"
                )
        except Exception as e:
            st.error(f"Error mostrando métricas globales: {e}")
    
    # Sección de análisis automático
    st.markdown("---")
    st.markdown("### 🎯 Análisis Rápido del Mercado")
    
    analysis_cols = st.columns(3)
    
    with analysis_cols[0]:
        if st.button("🚀 Análisis General", use_container_width=True):
            if st.session_state.selected_ai and config_loaded:
                market_summary = create_market_summary(prices_data, fear_greed_data, market_data)
                
                # Prompt específico para análisis general
                general_prompt = f"""Como analista senior, dame un análisis completo del mercado crypto actual.

DATOS DEL MERCADO:
- Bitcoin: ${prices_data['bitcoin']['usd']:,.2f} ({prices_data['bitcoin']['usd_24h_change']:+.2f}%) si prices_data else 'N/A'
- Fear & Greed: {fear_greed_data['value']}/100 ({fear_greed_data['value_classification']}) si fear_greed_data else 'N/A'
- Market Cap Total: ${market_data['total_market_cap']['usd']/1e12:.2f}T si market_data else 'N/A'

INCLUIR:
1. Resumen ejecutivo del sentiment actual
2. Análisis de Bitcoin como líder del mercado
3. Oportunidades en altcoins principales (ETH, SOL, ADA)
4. Catalyzadores macro relevantes
5. Niveles clave a monitorear esta semana

Respuesta en español, formato markdown, máximo 600 palabras."""
                
                with st.spinner("🔍 Generando análisis general..."):
                    try:
                        analysis = analyze_market(
                            general_prompt,
                            market_summary,
                            provider=st.session_state.selected_ai
                        )
                        
                        st.session_state.chat_history.append({
                            "role": "user",
                            "message": "🚀 Análisis General del Mercado"
                        })
                        
                        st.session_state.chat_history.append({
                            "role": "ai",
                            "message": analysis,
                            "provider": st.session_state.selected_ai
                        })
                    except Exception as e:
                        st.error(f"Error en análisis general: {e}")
                        
                st.rerun()
            else:
                st.error("❌ No hay IAs disponibles")
    
    with analysis_cols[1]:
        if st.button("⚖️ Gestión de Riesgo", use_container_width=True):
            if st.session_state.selected_ai and config_loaded:
                market_summary = create_market_summary(prices_data, fear_greed_data, market_data)
                
                # Prompt específico para gestión de riesgo
                risk_prompt = f"""Como especialista en gestión de riesgo para crypto trading, necesito recomendaciones específicas.

CONTEXTO ACTUAL:
- Volatilidad del mercado: {"Alta" if fear_greed_data and int(fear_greed_data['value']) < 40 else "Media"}
- Bitcoin dominancia: {market_data['market_cap_percentage']['btc']:.1f}% si market_data else 'N/A'
- Sentiment general: {fear_greed_data['value_classification'] if fear_greed_data else 'N/A'}

PROPORCIONAR:
1. Sizing de posición recomendado para diferentes profiles de riesgo
2. Niveles de stop-loss específicos para BTC, ETH, SOL
3. Estrategias de diversificación actuales  
4. Hedging contra caídas del mercado
5. Métricas de riesgo a monitorear diariamente
6. Gestión de emociones en el mercado actual

Incluye ejemplos numéricos prácticos. Respuesta en español, formato markdown."""
                
                with st.spinner("⚖️ Analizando gestión de riesgo..."):
                    try:
                        analysis = analyze_market(
                            risk_prompt,
                            market_summary,
                            provider=st.session_state.selected_ai
                        )
                        
                        st.session_state.chat_history.append({
                            "role": "user",
                            "message": "⚖️ Análisis de Gestión de Riesgo"
                        })
                        
                        st.session_state.chat_history.append({
                            "role": "ai", 
                            "message": analysis,
                            "provider": st.session_state.selected_ai
                        })
                    except Exception as e:
                        st.error(f"Error en análisis de riesgo: {e}")
                        
                st.rerun()
            else:
                st.error("❌ No hay IAs disponibles")
    
    with analysis_cols[2]:
        if st.button("🎯 Puntos de Entrada", use_container_width=True):
            if st.session_state.selected_ai and config_loaded:
                market_summary = create_market_summary(prices_data, fear_greed_data, market_data)
                
                # Prompt específico para puntos de entrada
                entry_prompt = f"""Como trader técnico especializado, identifica puntos de entrada óptimos.

PRECIOS ACTUALES:
- BTC: ${prices_data['bitcoin']['usd']:,.2f} ({prices_data['bitcoin']['usd_24h_change']:+.2f}%)
- ETH: ${prices_data['ethereum']['usd']:,.2f} ({prices_data['ethereum']['usd_24h_change']:+.2f}%)
- SOL: ${prices_data['solana']['usd']:.2f} ({prices_data['solana']['usd_24h_change']:+.2f}%)

ASSET PRINCIPAL: {st.session_state.current_symbol.replace('-USD', '')}
TIMEFRAME: {st.session_state.current_timeframe}

ANÁLISIS REQUERIDO:
1. Niveles de entrada específicos para {st.session_state.current_symbol.replace('-USD', '')}
2. Confirmaciones técnicas necesarias antes de entrar
3. Múltiples timeframes (1h, 4h, 1d) para confluencia
4. Volumen y momentum requeridos
5. Puntos de invalidación de la operación
6. Targets de profit taking (TP1, TP2, TP3)
7. Timing óptimo para la entrada

Incluye niveles de precio exactos y condiciones específicas. Respuesta técnica detallada."""
                
                with st.spinner("🎯 Identificando puntos de entrada..."):
                    try:
                        analysis = analyze_market(
                            entry_prompt,
                            market_summary,
                            provider=st.session_state.selected_ai
                        )
                        
                        st.session_state.chat_history.append({
                            "role": "user",
                            "message": f"🎯 Puntos de Entrada para {st.session_state.current_symbol.replace('-USD', '')}"
                        })
                        
                        st.session_state.chat_history.append({
                            "role": "ai",
                            "message": analysis, 
                            "provider": st.session_state.selected_ai
                        })
                    except Exception as e:
                        st.error(f"Error identificando puntos de entrada: {e}")
                        
                st.rerun()
            else:
                st.error("❌ No hay IAs disponibles")
    
    # Footer con información
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 1.5rem;'>
        <strong>📈 Trading Assistant Pro</strong> • Desarrollado con Python & Streamlit<br>
        🤖 Powered by Claude, GPT-4 & Gemini • 📊 Datos en tiempo real<br>
        <small>⚠️ <em>Este análisis es solo educativo y no constituye consejo financiero. Siempre haz tu propia investigación (DYOR).</em></small>
    </div>
    """, unsafe_allow_html=True)
    
    # Debug info (solo si DEBUG está habilitado)
    if config_loaded and hasattr(Config, 'DEBUG') and Config.DEBUG:
        with st.expander("🐛 Debug Information"):
            try:
                debug_info = {
                    "Session State": {
                        "Current Symbol": st.session_state.current_symbol,
                        "Timeframe": st.session_state.current_timeframe, 
                        "Selected AI": st.session_state.selected_ai,
                        "Chat History Length": len(st.session_state.chat_history),
                        "Show AI Comparison": st.session_state.show_ai_comparison
                    },
                    "Market Data Status": {
                        "Prices Data": bool(prices_data),
                        "Market Data": bool(market_data),
                        "Fear & Greed Data": bool(fear_greed_data)
                    },
                    "AI Status": get_ai_stats() if config_loaded else "Not loaded",
                    "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "Config Loaded": config_loaded
                }
                st.json(debug_info)
            except Exception as e:
                st.error(f"Error en debug info: {e}")

# Funciones adicionales para manejo de errores y compatibilidad
def safe_import_check():
    """Verifica que las importaciones críticas estén disponibles"""
    missing_modules = []
    
    try:
        import streamlit
    except ImportError:
        missing_modules.append("streamlit")
    
    try:
        import plotly
    except ImportError:
        missing_modules.append("plotly")
    
    try:
        import pandas
    except ImportError:
        missing_modules.append("pandas")
    
    try:
        import requests
    except ImportError:
        missing_modules.append("requests")
    
    return missing_modules

def display_installation_guide():
    """Muestra guía de instalación para Python 3.8"""
    st.markdown("""
    ## 🛠️ Guía de Instalación para Python 3.8
    
    ### Instalar dependencias compatibles:
    
    ```bash
    # Versiones específicas para Python 3.8
    pip install streamlit==1.25.0
    pip install plotly==5.15.0
    pip install pandas==2.0.3
    pip install numpy==1.24.4
    pip install requests==2.31.0
    pip install python-dotenv==1.0.0
    
    # IAs (opcional, para funcionalidad completa)
    pip install openai==0.28.1
    pip install anthropic==0.3.11
    pip install google-generativeai==0.3.1
    ```
    
    ### Ejecutar la aplicación:
    ```bash
    python -m streamlit run app.py
    ```
    
    ### Si tienes problemas:
    1. Verifica tu versión de Python: `python --version`
    2. Actualiza pip: `python -m pip install --upgrade pip`
    3. Considera actualizar a Python 3.10+ para mejor compatibilidad
    """)

def get_system_info():
    """Obtiene información del sistema para debugging"""
    import platform
    
    info = {
        "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Platform": platform.system(),
        "Platform Version": platform.version(),
        "Architecture": platform.machine(),
        "Streamlit Version": st.__version__ if hasattr(st, '__version__') else "Unknown"
    }
    
    return info

# Verificación inicial al cargar el módulo
if __name__ == "__main__":
    missing = safe_import_check()
    if missing:
        print(f"⚠️ Módulos faltantes: {', '.join(missing)}")
        print("Ejecuta la instalación de dependencias antes de continuar.")
    else:
        print("✅ Todas las dependencias básicas están disponibles")
        
    # Ejecutar la aplicación
    try:
        main()
    except Exception as critical_error:
        print(f"❌ Error crítico: {critical_error}")
        
        # Mostrar información del sistema si hay error crítico
        try:
            import streamlit as st_fallback
            st_fallback.error(f"Error crítico en la aplicación: {critical_error}")
            
            with st_fallback.expander("🔍 Información del Sistema"):
                system_info = get_system_info()
                st_fallback.json(system_info)
                
            with st_fallback.expander("📋 Guía de Solución"):
                display_installation_guide()
                
        except:
            # Si ni siquiera Streamlit funciona, mostrar en consola
            print("""
            ❌ ERROR CRÍTICO - La aplicación no puede iniciarse
            
            Soluciones recomendadas:
            
            1. Para Python 3.8, instala versiones específicas:
               pip install streamlit==1.25.0 plotly==5.15.0 pandas==2.0.3
            
            2. O actualiza Python a 3.10+:
               - Descarga desde https://python.org
               - Reinstala todas las librerías
               
            3. Verifica que todos los archivos estén presentes:
               - config.py
               - ai_providers/ (carpeta completa)
               - .env (con tus API keys)
            """)

# Meta información del archivo
__version__ = "2.0.0"
__author__ = "Trading Assistant Pro"
__description__ = "Aplicación de análisis de trading con IA múltiple compatible con Python 3.8+"
__python_requires__ = ">=3.8"

# Configuraciones adicionales para producción
PRODUCTION_CONFIG = {
    "debug": False,
    "host": "0.0.0.0",
    "port": 8501,
    "max_upload_size": 200,  # MB
    "theme": {
        "primaryColor": "#3b82f6",
        "backgroundColor": "#0f172a",
        "secondaryBackgroundColor": "#1e293b",
        "textColor": "#f8fafc"
    }
}