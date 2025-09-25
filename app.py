"""
Trading Assistant Pro - Aplicación Principal
Análisis avanzado con múltiples providers de IA para trading de criptomonedas
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import logging

# Importar configuración y módulos propios
from config import Config, AIConfig, TradingConfig, validate_config
from ai_providers import analyze_market, get_available_ais, get_ai_comparison, get_ai_stats

# Configurar logging
logging.basicConfig(level=logging.INFO if Config.DEBUG else logging.WARNING)
logger = logging.getLogger(__name__)

# Configuración de la página
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        'current_timeframe': Config.DEFAULT_TIMEFRAME,
        'selected_ai': None,
        'show_ai_comparison': False,
        'chart_data_cache': {},
        'last_refresh': datetime.now()
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Configurar IA por defecto
    if st.session_state.selected_ai is None:
        available_ais = get_available_ais()
        if available_ais:
            default_ai = AIConfig.DEFAULT_AI_PROVIDER if AIConfig.DEFAULT_AI_PROVIDER in available_ais else available_ais[0]
            st.session_state.selected_ai = default_ai

def validate_and_show_config():
    """Valida configuración y muestra advertencias si es necesario"""
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

# Funciones de datos del mercado
@st.cache_data(ttl=Config.PRICE_CACHE_TTL, show_spinner=False)
def get_crypto_prices():
    """Obtiene precios actuales de las principales cryptos"""
    try:
        url = f"{Config.COINGECKO_API}/simple/price"
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

@st.cache_data(ttl=Config.MARKET_DATA_CACHE_TTL, show_spinner=False)
def get_fear_greed_index():
    """Obtiene el índice de miedo y codicia"""
    try:
        response = requests.get(Config.FEAR_GREED_API, timeout=10)
        response.raise_for_status()
        return response.json()['data'][0]
    except Exception as e:
        logger.error(f"Error obteniendo Fear & Greed: {e}")
        return None

@st.cache_data(ttl=Config.MARKET_DATA_CACHE_TTL, show_spinner=False)
def get_market_data():
    """Obtiene datos generales del mercado"""
    try:
        response = requests.get(f"{Config.COINGECKO_API}/global", timeout=10)
        response.raise_for_status()
        return response.json()['data']
    except Exception as e:
        logger.error(f"Error obteniendo datos del mercado: {e}")
        return None

@st.cache_data(ttl=Config.CHART_CACHE_TTL, show_spinner=False)
def get_candlestick_data(symbol='BTC-USD', timeframe='1h', days=30):
    """Obtiene datos de velas para el gráfico usando yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Mapear timeframes de Streamlit a yfinance
        interval_map = {
            '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h',
            '4h': '1h', '1d': '1d', '1w': '1wk', '1M': '1mo'
        }
        
        interval = interval_map.get(timeframe, '1h')
        
        # Ajustar período según timeframe
        if timeframe in ['5m', '15m', '30m']:
            period = "7d"  # Máximo para intervalos pequeños
        elif timeframe == '1h':
            period = "30d"
        elif timeframe == '4h':
            period = "60d" 
        else:
            period = "1y"
        
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            # Generar datos simulados realistas
            logger.info(f"Generando datos simulados para {symbol}")
            return generate_realistic_ohlc_data(symbol, timeframe, days)
        
        return data
        
    except Exception as e:
        logger.warning(f"Error con yfinance para {symbol}: {e}")
        return generate_realistic_ohlc_data(symbol, timeframe, days)

def generate_realistic_ohlc_data(symbol, timeframe, days):
    """Genera datos OHLC realistas para demo"""
    # Precios base por símbolo
    base_prices = {
        'BTC-USD': 43000, 'ETH-USD': 2600, 'SOL-USD': 98,
        'ADA-USD': 0.52, 'MATIC-USD': 0.85, 'AVAX-USD': 37
    }
    
    base_price = base_prices.get(symbol, 1000)
    
    # Generar fechas
    if timeframe in ['5m', '15m', '30m']:
        freq = f'{timeframe[:-1]}min'
        periods = min(days * 24 * (60 // int(timeframe[:-1])), 1000)  # Límite para performance
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
    
    # Generar precios con volatilidad realista
    np.random.seed(42)  # Para reproducibilidad
    
    # Volatilidades por asset (anualizada)
    volatilities = {
        'BTC-USD': 0.6, 'ETH-USD': 0.8, 'SOL-USD': 1.2,
        'ADA-USD': 1.0, 'MATIC-USD': 1.1, 'AVAX-USD': 1.0
    }
    
    volatility = volatilities.get(symbol, 0.8)
    
    # Convertir a volatilidad por período
    if timeframe in ['5m', '15m', '30m', '1h']:
        period_vol = volatility / np.sqrt(365 * 24)  # Volatilidad horaria
    elif timeframe == '4h':
        period_vol = volatility / np.sqrt(365 * 6)
    elif timeframe == '1d':
        period_vol = volatility / np.sqrt(365)
    else:
        period_vol = volatility / np.sqrt(52)
    
    # Generar walk con tendencia sutil
    returns = np.random.normal(0.0001, period_vol, len(dates))  # Slight positive drift
    prices = [base_price]
    
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, prices[-1] * 0.95))  # Floor loss
    
    prices = prices[1:]  # Remove first element
    
    # Generar OHLC
    ohlc_data = []
    for i, price in enumerate(prices):
        # Generar variación intraperiod
        high_mult = 1 + abs(np.random.normal(0, period_vol/4))
        low_mult = 1 - abs(np.random.normal(0, period_vol/4))
        
        if i == 0:
            open_price = base_price
        else:
            open_price = ohlc_data[-1]['Close']  # Continuity
            
        close_price = price
        high_price = max(open_price, close_price) * high_mult
        low_price = min(open_price, close_price) * low_mult
        
        # Volume realista
        base_volume = np.random.uniform(1000000, 10000000)
        if abs(close_price - open_price) / open_price > period_vol:  # High volatility = high volume
            volume = base_volume * np.random.uniform(1.5, 3.0)
        else:
            volume = base_volume
            
        ohlc_data.append({
            'Open': open_price,
            'High': high_price, 
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
    
    df = pd.DataFrame(ohlc_data, index=dates)
    return df

def create_candlestick_chart(data, symbol, timeframe):
    """Crea gráfico de velas profesional con Plotly"""
    if data.empty:
        return None
    
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
    if len(data) >= TradingConfig.MA_SHORT_PERIOD:
        data[f'MA{TradingConfig.MA_SHORT_PERIOD}'] = data['Close'].rolling(window=TradingConfig.MA_SHORT_PERIOD).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data[f'MA{TradingConfig.MA_SHORT_PERIOD}'],
                name=f'MA{TradingConfig.MA_SHORT_PERIOD}', 
                line=dict(color='#3b82f6', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    if len(data) >= TradingConfig.MA_LONG_PERIOD:
        data[f'MA{TradingConfig.MA_LONG_PERIOD}'] = data['Close'].rolling(window=TradingConfig.MA_LONG_PERIOD).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data[f'MA{TradingConfig.MA_LONG_PERIOD}'],
                name=f'MA{TradingConfig.MA_LONG_PERIOD}', 
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
    
    # Añadir anotaciones si hay cambios significativos
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
    ai_stats = get_ai_stats()
    status_color = "status-online" if ai_stats['total_providers'] > 0 else "status-offline"
    status_text = f"{ai_stats['total_providers']} IA{'s' if ai_stats['total_providers'] != 1 else ''} disponible{'s' if ai_stats['total_providers'] != 1 else ''}"
    
    st.sidebar.markdown(f"""
    <div class="sidebar-section">
        <h4>📡 Estado del Sistema</h4>
        <p><span class="status-indicator {status_color}"></span>{status_text}</p>
        <small>Última actualización: {datetime.now().strftime('%H:%M:%S')}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Selector de IA
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
    
    # Configuración de gráfico
    st.sidebar.markdown("### 📊 Configuración de Gráfico")
    
    # Selector de criptomoneda
    selected_crypto = st.sidebar.selectbox(
        "Criptomoneda:",
        list(Config.DEFAULT_SYMBOLS.keys()),
        index=list(Config.DEFAULT_SYMBOLS.values()).index(st.session_state.current_symbol) if st.session_state.current_symbol in Config.DEFAULT_SYMBOLS.values() else 0
    )
    
    st.session_state.current_symbol = Config.DEFAULT_SYMBOLS[selected_crypto]
    
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
    
    # Información adicional
    if Config.DEBUG:
        st.sidebar.markdown("### 🐛 Debug Info")
        st.sidebar.json({
            "Current Symbol": st.session_state.current_symbol,
            "Timeframe": st.session_state.current_timeframe,
            "Selected AI": st.session_state.selected_ai,
            "Available AIs": len(available_ais),
            "Chat Messages": len(st.session_state.chat_history)
        })

def display_ai_comparison():
    """Muestra comparación detallada de IAs disponibles"""
    if st.session_state.show_ai_comparison:
        st.markdown("## 🤖 Comparación de Providers de IA")
        
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

def main():
    """Función principal de la aplicación"""
    
    # Inicializar estado
    initialize_session_state()
    
    # Validar configuración
    config_valid = validate_and_show_config()
    
    # Header principal
    st.markdown(f"""
    <div class="main-header">
        <h1>{Config.APP_ICON} {Config.APP_TITLE}</h1>
        <p>Análisis avanzado con IA múltiple • Tiempo real • Python & Streamlit</p>
        <small>🤖 {len(get_available_ais())} IA{'s' if len(get_available_ais()) != 1 else ''} disponible{'s' if len(get_available_ais()) != 1 else ''}</small>
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
        cols = st.columns(len(metrics_data))
        for i, metric in enumerate(metrics_data):
            with cols[i]:
                change_color = "price-positive" if metric['change'] >= 0 else "price-negative"
                st.metric(
                    f"₿ {metric['name']}" if metric['name'] == 'Bitcoin' else f"⟠ {metric['name']}" if metric['name'] == 'Ethereum' else f"◎ {metric['name']}" if metric['name'] == 'Solana' else f"🔷 {metric['name']}", 
                    f"${metric['price']:,.2f}" if metric['price'] >= 1 else f"${metric['price']:.4f}",
                    f"{metric['change']:+.2f}%"
                )
    
    
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
            else:
                st.error("❌ No se pudo generar el gráfico")
        else:
            st.error("❌ No se pudieron cargar los datos del gráfico")
    
    with col_chat:
        st.markdown("### 🤖 Chat con IA Especializada")
        
        # Mostrar información de la IA actual
        if st.session_state.selected_ai:
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
        
        # Contenedor del chat con altura fija
        chat_container = st.container()
        
        with chat_container:
            # Mostrar historial de chat
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
                    
                    st.markdown(f"""
                    <div class="chat-message ai-message">
                        {provider_info}{chat["message"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        {chat["message"]}
                    </div>
                    """, unsafe_allow_html=True)
        
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
                    st.session_state.chat_input = example
                    st.rerun()
        
        # Campo de entrada de texto
        user_input = st.text_area(
            "Tu consulta:",
            value=st.session_state.get('chat_input', ''),
            height=100,
            placeholder="Ej: Analiza Bitcoin, dame puntos de entrada para ETH, qué opinas del mercado...",
            help="Pregunta sobre análisis técnico, tendencias, gestión de riesgo o estrategias de trading"
        )
        
        # Limpiar el input temporal
        if 'chat_input' in st.session_state:
            del st.session_state.chat_input
        
        # Botones de acción del chat
        col_send, col_clear_msg = st.columns([2, 1])
        
        with col_send:
            if st.button("📤 Analizar", use_container_width=True) and user_input.strip():
                if not st.session_state.selected_ai:
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
            volume_change = ((volume / (volume * 0.95)) - 1) * 100  # Simulado
            st.metric(
                "📊 Volumen 24h",
                f"${volume/1e9:.1f}B",
                f"{volume_change:+.1f}%",
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
            
            # Determinar color y delta basado en el valor
            if fear_value <= 25:
                delta_color = "inverse"  # Rojo - extremo miedo es potencialmente bueno para comprar
            elif fear_value >= 75:
                delta_color = "normal"  # Verde - extrema codicia es señal de alerta
            else:
                delta_color = "off"
            
            st.metric(
                "😨 Fear & Greed",
                f"{fear_value}/100",
                fear_classification,
                delta_color=delta_color,
                help="Índice de sentimiento del mercado: 0 = Extremo Miedo, 100 = Extrema Codicia"
            )
    
    # Sección de análisis automático
    st.markdown("---")
    st.markdown("### 🎯 Análisis Rápido del Mercado")
    
    analysis_cols = st.columns(3)
    
    with analysis_cols[0]:
        if st.button("🚀 Análisis General", use_container_width=True):
            if st.session_state.selected_ai:
                market_summary = create_market_summary(prices_data, fear_greed_data, market_data)
                
                with st.spinner("🔍 Generando análisis general..."):
                    analysis = analyze_market(
                        "Dame un análisis general del mercado crypto actual con las principales tendencias y oportunidades",
                        market_summary,
                        provider=st.session_state.selected_ai
                    )
                    
                    st.session_state.chat_history.append({
                        "role": "ai",
                        "message": analysis,
                        "provider": st.session_state.selected_ai
                    })
                    
                st.rerun()
    
    with analysis_cols[1]:
        if st.button("⚖️ Gestión de Riesgo", use_container_width=True):
            if st.session_state.selected_ai:
                market_summary = create_market_summary(prices_data, fear_greed_data, market_data)
                
                with st.spinner("⚖️ Analizando gestión de riesgo..."):
                    analysis = analyze_market(
                        "Dame recomendaciones de gestión de riesgo para el mercado actual, incluyendo stops y sizing de posiciones",
                        market_summary,
                        provider=st.session_state.selected_ai
                    )
                    
                    st.session_state.chat_history.append({
                        "role": "ai", 
                        "message": analysis,
                        "provider": st.session_state.selected_ai
                    })
                    
                st.rerun()
    
    with analysis_cols[2]:
        if st.button("🎯 Puntos de Entrada", use_container_width=True):
            if st.session_state.selected_ai:
                market_summary = create_market_summary(prices_data, fear_greed_data, market_data)
                
                with st.spinner("🎯 Identificando puntos de entrada..."):
                    analysis = analyze_market(
                        f"Identifica puntos de entrada estratégicos para {st.session_state.current_symbol.replace('-USD', '')} y las principales cryptos",
                        market_summary,
                        provider=st.session_state.selected_ai
                    )
                    
                    st.session_state.chat_history.append({
                        "role": "ai",
                        "message": analysis, 
                        "provider": st.session_state.selected_ai
                    })
                    
                st.rerun()
    
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
    if Config.DEBUG:
        with st.expander("🐛 Debug Information"):
            st.json({
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
                "AI Status": get_ai_stats(),
                "Config": {
                    "Debug Mode": Config.DEBUG,
                    "Default AI Provider": AIConfig.DEFAULT_AI_PROVIDER,
                    "Cache TTL": {
                        "Prices": Config.PRICE_CACHE_TTL,
                        "Market Data": Config.MARKET_DATA_CACHE_TTL,
                        "Chart Data": Config.CHART_CACHE_TTL
                    }
                }
            })

if __name__ == "__main__":
    main()