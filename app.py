"""
Trading Assistant Pro - Aplicación Principal ACTUALIZADA
Análisis avanzado con múltiples providers de IA + Utils integrados
VERSIÓN COMPATIBLE CON PYTHON 3.8 + Indicadores Técnicos + Sentiment Analysis
"""

import streamlit as st
import html
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

from utils.binance_data import (
    get_all_symbols, 
    get_ticker_data, 
    get_top_movers, 
    search_symbol,
    get_chart_data,
    get_realtime_price
)

# Importar módulos propios con manejo de errores (ANTES de cualquier comando de Streamlit)
try:
    from config import Config, AIConfig, TradingConfig, validate_config
    from ai_providers import analyze_market, get_available_ais, get_ai_comparison, get_ai_stats
    from utils.technical_indicators import TechnicalIndicators, MarketAnalyzer, format_analysis_for_ai
    from utils.news_sentiment import NewsAnalyzer, SentimentAggregator, format_sentiment_for_ai
    config_loaded = True
    utils_loaded = True
except ImportError as e:
    config_loaded = False
    utils_loaded = False
    import_error = str(e)
except Exception as e:
    config_loaded = False
    utils_loaded = False
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
    
if not utils_loaded:
    st.warning("⚠️ Los módulos de análisis técnico y sentiment no están disponibles. Funcionalidad limitada.")

# CSS personalizado mejorado con nuevos estilos para indicadores técnicos
# CSS personalizado mejorado - TEMA OSCURO BINANCE
st.markdown("""
<style>
    /* Tema oscuro completo */
    .stApp {
        background-color: #000000;
    }
    
    .main .block-container {
        background-color: #000000;
        padding-top: 2rem;
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #f0b90b, #f8d12f);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        text-align: center;
        color: #000000;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(240, 185, 11, 0.3);
    }
    
    /* Tabs para Spot/Futures */
    .market-tabs {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: #1a1a1a;
        border-radius: 8px;
    }
    
    .market-tab {
        padding: 0.5rem 1.5rem;
        background: #2b2b2b;
        border: 1px solid #3a3a3a;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s;
        color: #b0b0b0;
    }
    
    .market-tab.active {
        background: #f0b90b;
        color: #000000;
        border-color: #f0b90b;
    }
    
    .market-tab:hover {
        background: #3a3a3a;
        border-color: #f0b90b;
    }
    
    /* Buscador de símbolos */
    .symbol-search {
        background: #1a1a1a;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Lista de símbolos */
    .symbol-list {
        max-height: 400px;
        overflow-y: auto;
        background: #1a1a1a;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .symbol-item {
        padding: 0.8rem;
        background: #2b2b2b;
        border: 1px solid #3a3a3a;
        border-radius: 6px;
        margin: 0.3rem 0;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .symbol-item:hover {
        background: #3a3a3a;
        border-color: #f0b90b;
    }
    
    /* Métricas */
    .metric-card {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #3a3a3a;
        text-align: center;
    }
    
    div[data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 1px solid #3a3a3a;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Precios */
    .price-positive {
        color: #0ecb81 !important;
    }
    
    .price-negative {
        color: #f6465d !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #3a3a3a;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #1a1a1a;
    }
    
    /* Inputs y selectboxes */
    .stSelectbox > div > div,
    .stTextInput > div > div > input {
        background-color: #2b2b2b;
        color: #e0e0e0;
        border: 1px solid #3a3a3a;
    }
    
    /* Chat */
    .chat-container {
        height: 500px;
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background-color: #1a1a1a;
        border-radius: 8px;
        border: 1px solid #3a3a3a;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        word-wrap: break-word;
    }
    
    .user-message {
        background: linear-gradient(135deg, #f0b90b, #f8d12f);
        color: #000000;
        margin-left: 15%;
    }
    
    .ai-message {
        background: #2b2b2b;
        color: #e0e0e0;
        margin-right: 15%;
        border-left: 3px solid #f0b90b;
    }
    
    /* Botones */
    .stButton > button {
        background-color: #f0b90b;
        color: #000000;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #f8d12f;
        box-shadow: 0 4px 12px rgba(240, 185, 11, 0.4);
    }
    
    /* Scrollbar personalizado */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3a3a3a;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #f0b90b;
    }
    
    /* Texto general */
    .stMarkdown, p, span, div {
        color: #e0e0e0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Inicializa el estado de la sesión con nuevas variables para utils"""
    defaults = {
        'chat_history': [
            {
                "role": "ai", 
                "message": (
                "👋 **¡Bienvenido a Trading Assistant Pro!**\n\n"       
                "Soy tu asistente especializado en criptomonedas con:\n\n"
                "• **Claude, GPT-4 & Gemini** - Múltiples IAs especializadas\n"
                "• **Análisis técnico avanzado** - RSI, MACD, Bollinger Bands\n"
                "• **Análisis de sentiment** - Noticias y Fear & Greed\n"
                "• **Gráficos profesionales** - Datos en tiempo real\n\n"
                "¿Qué te gustaría analizar hoy?"
               
                ),   
                 "provider": "system"
            }
        ],
        'current_symbol': 'BTC-USD',
        'current_timeframe': '1h',
        'selected_ai': None,
        'show_ai_comparison': False,
        'show_technical_analysis': True,
        'show_sentiment_analysis': True,
        'chart_data_cache': {},
        'technical_analysis_cache': {},
        'sentiment_cache': {},
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

def setup_auto_refresh():
    """Configura auto-refresh para actualización en tiempo real"""
    if 'auto_refresh_enabled' not in st.session_state:
        st.session_state.auto_refresh_enabled = False
    
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 5  # segundos

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
                # Anthropic (Claude) - Análisis profundo  
                ANTHROPIC_API_KEY=sk-ant-your-claude-key-here
                
                # Google (Gemini) - Rápido y económico
                GEMINI_API_KEY=your-gemini-key-here
                
                # OpenAI (GPT-4) - Versátil y balanceado
                OPENAI_API_KEY=sk-your-openai-key-here
                ```
                
                **Dónde obtener las keys:**
                - Claude: https://console.anthropic.com/
                - Gemini: https://ai.google.dev/
                - OpenAI: https://platform.openai.com/api-keys
                """)
            
            return False
        return True
    except Exception as e:
        st.error(f"❌ Error validando configuración: {e}")
        return False

# Funciones de datos del mercado MEJORADAS
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
def get_candlestick_data(symbol='BTC-USD', timeframe='1h', days=180, market_type='spot'):
    """Obtiene datos de velas para el gráfico usando Binance API (Spot o Futuros)"""
    try:
        # Mapeo de símbolos (Spot)
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

        # Endpoint según mercado
        if market_type == 'spot':
            base_url = "https://api.binance.com/api/v3/klines"
        else:  # futures
            base_url = "https://fapi.binance.com/fapi/v1/klines"

        # Llamada a la API
        limit = min(days * 24, 1000)
        params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Conversión a DataFrame (igual que ya lo tenías)
        df = pd.DataFrame(data, columns=[
            "OpenTime", "Open", "High", "Low", "Close", "Volume",
            "CloseTime", "QuoteAssetVolume", "Trades",
            "TakerBuyBase", "TakerBuyQuote", "Ignore"
        ])
        df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms")
        df.set_index("OpenTime", inplace=True)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        logger.error(f"Error obteniendo datos de Binance ({market_type}) para {symbol}: {e}")
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
            new_price = max(new_price, prices[-1] * 0.95)
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

# NUEVA FUNCIÓN: Análisis técnico integrado
@st.cache_data(ttl=120, show_spinner=False)
def get_technical_analysis(symbol, chart_data):
    """Obtiene análisis técnico completo usando los utils"""
    if not utils_loaded or chart_data.empty:
        return None
    
    try:
        analyzer = MarketAnalyzer()
        analysis = analyzer.comprehensive_analysis(chart_data)
        
        if 'error' not in analysis:
            # Generar señales
            signals = analyzer.generate_signals(analysis)
            analysis['signals'] = signals
        
        return analysis
    except Exception as e:
        logger.error(f"Error en análisis técnico: {e}")
        return None

# NUEVA FUNCIÓN: Análisis de sentiment integrado
@st.cache_data(ttl=600, show_spinner=False)  # Cache por 10 minutos
def get_sentiment_analysis(symbol):
    """Obtiene análisis de sentiment usando los utils"""
    if not utils_loaded:
        return None
    
    try:
        # Configurar NewsAnalyzer (sin API key por ahora, usa fuentes gratuitas)
        news_analyzer = NewsAnalyzer(news_api_key=AIConfig.NEWS_API_KEY)
        sentiment_aggregator = SentimentAggregator()
        
        # Obtener análisis de sentiment general
        sentiment_data = sentiment_aggregator.get_market_sentiment(news_analyzer)
        
        return sentiment_data
    except Exception as e:
        logger.error(f"Error en análisis de sentiment: {e}")
        return None

def create_candlestick_chart(data, symbol, timeframe, technical_analysis=None):
    """Crea gráfico de velas estilo Binance con máxima interactividad"""
    if data.empty:
        return None
    
    try:
        # Crear subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} - {timeframe.upper()}', 'Volumen', 'RSI'),
            row_heights=[0.7, 0.15, 0.15]
        )
        
        # Velas con colores más intensos tipo Binance
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Precio",
                increasing_line_color='#0ecb81',  # Verde Binance
                decreasing_line_color='#f6465d',  # Rojo Binance
                increasing_fillcolor='#0ecb81',
                decreasing_fillcolor='#f6465d',
                increasing_line_width=1.5,
                decreasing_line_width=1.5
            ),
            row=1, col=1
        )
        
        # Medias móviles
        if len(data) >= 20:
            data_copy = data.copy()
            data_copy['MA20'] = data_copy['Close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=data_copy.index, 
                    y=data_copy['MA20'],
                    name='MA20', 
                    line=dict(color='#f0b90b', width=1.5),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        if len(data) >= 50:
            data_copy['MA50'] = data_copy['Close'].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(
                    x=data_copy.index, 
                    y=data_copy['MA50'],
                    name='MA50', 
                    line=dict(color='#2962ff', width=1.5),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        # Bollinger Bands si disponibles
        if technical_analysis and utils_loaded and 'bollinger' in technical_analysis:
            try:
                bb = technical_analysis['bollinger']
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=bb['upper'],
                        name='BB Upper',
                        line=dict(color='#787b86', width=1, dash='dash'),
                        opacity=0.5
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=bb['lower'],
                        name='BB Lower',
                        line=dict(color='#787b86', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(120, 123, 134, 0.1)',
                        opacity=0.5
                    ),
                    row=1, col=1
                )
            except Exception as e:
                logger.warning(f"Error añadiendo Bollinger Bands: {e}")
        
        # Volumen con colores
        colors = ['#0ecb81' if data['Close'].iloc[i] >= data['Open'].iloc[i] 
                  else '#f6465d' for i in range(len(data))]
        
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
        
        # RSI
        if technical_analysis and utils_loaded and len(data) >= 14:
            try:
                indicators = TechnicalIndicators()
                rsi_values = indicators.rsi(data['Close'])
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=rsi_values,
                        name="RSI",
                        line=dict(color='#f0b90b', width=2)
                    ),
                    row=3, col=1
                )
                
                fig.add_hline(y=70, line=dict(color='#f6465d', dash='dash', width=1), row=3, col=1)
                fig.add_hline(y=30, line=dict(color='#0ecb81', dash='dash', width=1), row=3, col=1)
                fig.add_hline(y=50, line=dict(color='#787b86', dash='dot', width=1), row=3, col=1)
            except Exception as e:
                logger.warning(f"Error añadiendo RSI: {e}")
        
        # Layout estilo Binance con máxima interactividad
        fig.update_layout(
            title={
                'text': f"<b>{symbol}</b> [{timeframe.upper()}]",
                'x': 0.5,
                'font': {'size': 20, 'color': '#ffffff'}
            },
            template="plotly_dark",
            height=750,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left", 
                x=0.01,
                bgcolor="rgba(26, 26, 26, 0.8)",
                font=dict(color='#e0e0e0')
            ),
            xaxis_rangeslider_visible=False,
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#e0e0e0'),
            margin=dict(l=70, r=70, t=80, b=60),
            
            # MÁXIMA INTERACTIVIDAD
            dragmode='pan',
            
            # Eje X con rango completo
            xaxis=dict(
                rangeslider=dict(visible=False),
                type="date",
                showspikes=True,
                spikecolor="#f0b90b",
                spikesnap="cursor",
                spikemode="across",
                gridcolor='#1a1a1a',
                showgrid=True,
                # CLAVE: Mostrar TODOS los datos
                range=[data.index.min(), data.index.max()],
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1H", step="hour", stepmode="backward"),
                        dict(count=4, label="4H", step="hour", stepmode="backward"),
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(count=7, label="7D", step="day", stepmode="backward"),
                        dict(count=30, label="1M", step="day", stepmode="backward"),
                        dict(step="all", label="Todo")
                    ]),
                    bgcolor="rgba(26, 26, 26, 0.9)",
                    activecolor="#f0b90b",
                    bordercolor="#3a3a3a",
                    font=dict(color='#e0e0e0')
                )
            ),
            
            # Eje Y con scroll para ajustar escala
            yaxis=dict(
                fixedrange=False,  # Permitir ajuste vertical
                showspikes=True,
                spikecolor="#f0b90b",
                spikesnap="cursor",
                spikemode="across",
                gridcolor='#1a1a1a',
                showgrid=True,
                side='right'  # Precios a la derecha como Binance
            ),
            yaxis2=dict(
                gridcolor='#1a1a1a',
                showgrid=True,
                side='right'
            ),
            yaxis3=dict(
                gridcolor='#1a1a1a',
                showgrid=True,
                range=[0, 100],
                side='right'
            ),
            
            # Barra de herramientas completa
            modebar=dict(
                bgcolor="rgba(26, 26, 26, 0.9)",
                color="#e0e0e0",
                activecolor="#f0b90b"
            ),
            
            hovermode='x unified',
            hoverdistance=100,
            spikedistance=1000
        )
        
        # Configurar ejes para permitir zoom con rueda del mouse
        fig.update_xaxes(gridcolor='#1a1a1a', showgrid=True)
        fig.update_yaxes(gridcolor='#1a1a1a', showgrid=True)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creando gráfico: {e}")
        return None
    


def create_market_summary_enhanced(prices_data, fear_greed_data, market_data, technical_analysis=None, sentiment_data=None, symbol=None):
    """Crea resumen MEJORADO del mercado para contexto de IA con análisis técnico y sentiment"""
    summary = {
        'prices': prices_data,
        'fear_greed': fear_greed_data,
        'global_data': market_data,
        'timestamp': datetime.now().isoformat()
    }
    
    # Agregar análisis técnico si está disponible
    if technical_analysis and utils_loaded:
        summary['technical_analysis'] = technical_analysis
    
    # Agregar análisis de sentiment si está disponible
    if sentiment_data and utils_loaded:
        summary['sentiment_analysis'] = sentiment_data
    
    # Agregar símbolo actual
    if symbol:
        summary['current_symbol'] = symbol
    
    return summary

def display_sidebar():
    """Renderiza la sidebar con controles mejorados estilo Binance"""
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
        
        utils_status = "✅ Activos" if utils_loaded else "❌ No disponibles"
        
        st.sidebar.markdown(f"""
        <div class="sidebar-section">
            <h4>📡 Estado del Sistema</h4>
            <p><span class="status-indicator {status_color}"></span>{status_text}</p>
            <p><small>📊 Utils: {utils_status}</small></p>
            <small>Última actualización: {datetime.now().strftime('%H:%M:%S')}</small>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"Error en estado: {e}")
    
    # Selector de IA
    if config_loaded:
        try:
            available_ais = get_available_ais()
            if available_ais:
                st.sidebar.markdown("### 🤖 Configuración de IA")
                
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
    
    # Configuración de análisis
    if utils_loaded:
        st.sidebar.markdown("### 📊 Configuración de Análisis")
        
        st.session_state.show_technical_analysis = st.sidebar.checkbox(
            "🔧 Análisis Técnico Avanzado",
            value=st.session_state.show_technical_analysis,
            help="Incluye RSI, MACD, Bollinger Bands, etc."
        )
        
        st.session_state.show_sentiment_analysis = st.sidebar.checkbox(
            "📰 Análisis de Sentiment",
            value=st.session_state.show_sentiment_analysis,
            help="Noticias, Fear & Greed, sentiment del mercado"
        )
    
    # === NUEVA SECCIÓN: Selección de Mercado y Activo ===
    st.sidebar.markdown("### 📈 Selección de Activo")
    
    # Tabs Spot/Futures
    market_type_options = ["Spot", "Futuros"]
    selected_market = st.sidebar.radio(
        "Tipo de Mercado:",
        market_type_options,
        index=0 if st.session_state.get('market_type', 'spot') == 'spot' else 1,
        horizontal=True
    )
    
    st.session_state.market_type = 'spot' if selected_market == "Spot" else 'futures'
    
    # Mostrar selector de símbolos en expander
    with st.sidebar.expander("🔍 Buscar Moneda", expanded=False):
        display_symbol_selector(st.session_state.market_type)
    
    # Mostrar símbolo actual
    current_symbol_display = st.session_state.current_symbol.replace('-USD', '').replace('USDT', '')
    st.sidebar.markdown(f"**Símbolo actual:** `{current_symbol_display}`")
    
    # Selector de timeframe
    st.sidebar.markdown("### ⏱️ Timeframe")
    
    timeframe_display = {
        '1m': '1 Minuto',
        '5m': '5 Minutos', 
        '15m': '15 Minutos', 
        '30m': '30 Minutos',
        '1h': '1 Hora', 
        '4h': '4 Horas', 
        '1d': '1 Día', 
        '1w': '1 Semana'
    }
    
    selected_timeframe_display = st.sidebar.selectbox(
        "Intervalo:",
        list(timeframe_display.values()),
        index=list(timeframe_display.keys()).index(st.session_state.current_timeframe) 
              if st.session_state.current_timeframe in timeframe_display else 4
    )
    
    # Mapear de vuelta al valor técnico
    reverse_timeframe = {v: k for k, v in timeframe_display.items()}
    st.session_state.current_timeframe = reverse_timeframe[selected_timeframe_display]
    
    # === AUTO-REFRESH PARA TIEMPO REAL ===
    st.sidebar.markdown("### ⚡ Actualización en Tiempo Real")
    
    auto_refresh = st.sidebar.checkbox(
        "Activar Auto-Refresh",
        value=st.session_state.get('auto_refresh_enabled', False),
        help="Actualiza el gráfico automáticamente cada X segundos"
    )
    st.session_state.auto_refresh_enabled = auto_refresh
    
    if auto_refresh:
        refresh_options = {
            3: "3 segundos (rápido)",
            5: "5 segundos (normal)",
            10: "10 segundos (lento)",
            30: "30 segundos (muy lento)"
        }
        
        selected_interval = st.sidebar.select_slider(
            "Intervalo de actualización:",
            options=list(refresh_options.keys()),
            value=st.session_state.get('refresh_interval', 5),
            format_func=lambda x: refresh_options[x]
        )
        st.session_state.refresh_interval = selected_interval
        
        st.sidebar.info(f"🔄 Actualizando cada {selected_interval}s")
    
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


def display_symbol_selector(market_type='spot'):
    """Selector de símbolos estilo Binance con búsqueda y filtros"""
    
    # Tabs para filtros
    filter_tab1, filter_tab2, filter_tab3 = st.tabs(["📊 Todos", "📈 Ganadores", "📉 Perdedores"])
    
    with filter_tab1:
        # Buscador
        search_query = st.text_input(
            "Buscar moneda",
            placeholder="Ej: BTC, ETH, SOL...",
            key=f"symbol_search_{market_type}"
        )
        
        # Obtener todos los símbolos
        all_symbols = get_all_symbols(market_type)
        
        if not all_symbols:
            st.warning("No se pudieron cargar los símbolos. Verifica tu conexión.")
            return
        
        # Filtrar por búsqueda
        if search_query:
            filtered_symbols = [
                s for s in all_symbols 
                if search_query.upper() in s['symbol'] or 
                   search_query.upper() in s['baseAsset']
            ]
        else:
            filtered_symbols = all_symbols[:50]  # Mostrar primeros 50
        
        # Obtener precios actuales
        ticker_data = get_ticker_data(market_type)
        
        # Mostrar lista de símbolos
        st.markdown(f"**{len(filtered_symbols)} símbolos encontrados**")
        
        for symbol_info in filtered_symbols:
            symbol = symbol_info['symbol']
            base_asset = symbol_info['baseAsset']
            
            # Obtener precio y cambio
            ticker = ticker_data.get(symbol, {})
            price = ticker.get('price', 0)
            change_24h = ticker.get('change_24h', 0)
            
            # Crear fila para cada símbolo
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown(f"**{base_asset}**")
            
            with col2:
                if price > 0:
                    change_color = "price-positive" if change_24h >= 0 else "price-negative"
                    price_display = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
                    st.markdown(f"<span class='{change_color}'>{price_display}</span>", 
                              unsafe_allow_html=True)
                else:
                    st.markdown("--")
            
            with col3:
                if price > 0:
                    change_color = "price-positive" if change_24h >= 0 else "price-negative"
                    st.markdown(f"<span class='{change_color}'>{change_24h:+.1f}%</span>", 
                              unsafe_allow_html=True)
            
            # Botón para seleccionar
            if st.button(f"Ver {base_asset}", key=f"select_{symbol}_{market_type}", use_container_width=True):
                if market_type == 'spot':
                    st.session_state.current_symbol = f"{base_asset}-USD"
                else:
                    st.session_state.current_symbol = symbol
                st.rerun()
            
            st.divider()
    
    with filter_tab2:
        # Top Ganadores
        try:
            gainers, _ = get_top_movers(market_type, limit=20)
            
            st.markdown("### 📈 Top 20 Ganadores 24h")
            
            if not gainers:
                st.info("No se pudieron cargar los ganadores")
                return
            
            for i, gainer in enumerate(gainers, 1):
                col1, col2, col3 = st.columns([1, 3, 2])
                
                with col1:
                    st.markdown(f"**#{i}**")
                
                with col2:
                    symbol_display = gainer['symbol'].replace('USDT', '')
                    st.markdown(f"**{symbol_display}**")
                    price_display = f"${gainer['price']:,.4f}" if gainer['price'] < 1 else f"${gainer['price']:,.2f}"
                    st.markdown(f"<span class='price-positive'>{price_display}</span>", 
                              unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"<span class='price-positive'>+{gainer['change_24h']:.2f}%</span>", 
                              unsafe_allow_html=True)
                
                if st.button(f"Ver", key=f"gainer_{i}_{market_type}", use_container_width=True):
                    st.session_state.current_symbol = gainer['symbol']
                    st.rerun()
                
                st.divider()
        except Exception as e:
            st.error(f"Error cargando ganadores: {e}")
    
    with filter_tab3:
        # Top Perdedores
        try:
            _, losers = get_top_movers(market_type, limit=20)
            
            st.markdown("### 📉 Top 20 Perdedores 24h")
            
            if not losers:
                st.info("No se pudieron cargar los perdedores")
                return
            
            for i, loser in enumerate(losers, 1):
                col1, col2, col3 = st.columns([1, 3, 2])
                
                with col1:
                    st.markdown(f"**#{i}**")
                
                with col2:
                    symbol_display = loser['symbol'].replace('USDT', '')
                    st.markdown(f"**{symbol_display}**")
                    price_display = f"${loser['price']:,.4f}" if loser['price'] < 1 else f"${loser['price']:,.2f}"
                    st.markdown(f"<span class='price-negative'>{price_display}</span>", 
                              unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"<span class='price-negative'>{loser['change_24h']:.2f}%</span>", 
                              unsafe_allow_html=True)
                
                if st.button(f"Ver", key=f"loser_{i}_{market_type}", use_container_width=True):
                    st.session_state.current_symbol = loser['symbol']
                    st.rerun()
                
                st.divider()
        except Exception as e:
            st.error(f"Error cargando perdedores: {e}")




def setup_auto_refresh():
    """Configura auto-refresh para actualización en tiempo real"""
    if 'auto_refresh_enabled' not in st.session_state:
        st.session_state.auto_refresh_enabled = False
    
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 5  # segundos



def display_technical_indicators(technical_analysis, symbol):
    """Muestra panel de indicadores técnicos"""
    if not technical_analysis or not utils_loaded:
        return
    
    st.markdown(f"### 🔧 Análisis Técnico - {symbol}")
    
    try:
        # Métricas principales en columnas
        tech_cols = st.columns(5)
        
        with tech_cols[0]:
            rsi = technical_analysis.get('rsi')
            if rsi:
                rsi_color = "#ef4444" if rsi > 70 else "#10b981" if rsi < 30 else "#6b7280"
                rsi_status = "Sobrecompra" if rsi > 70 else "Sobreventa" if rsi < 30 else "Neutral"
                st.markdown(f"""
                <div class="technical-indicator">
                    <strong>RSI (14)</strong><br>
                    <span style="color: {rsi_color}; font-size: 1.5em;">{rsi:.1f}</span><br>
                    <small>{rsi_status}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with tech_cols[1]:
            levels = technical_analysis.get('levels', {})
            if levels:
                distance_to_resistance = levels.get('distance_to_resistance', 0)
                color = "#ef4444" if distance_to_resistance < 5 else "#10b981"
                st.markdown(f"""
                <div class="technical-indicator">
                    <strong>Resistencia</strong><br>
                    <span style="color: {color}; font-size: 1.2em;">${levels.get('resistance', 0):.2f}</span><br>
                    <small>{distance_to_resistance:+.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
        
        with tech_cols[2]:
            if levels:
                distance_to_support = levels.get('distance_to_support', 0)
                color = "#10b981" if distance_to_support > 5 else "#ef4444"
                st.markdown(f"""
                <div class="technical-indicator">
                    <strong>Soporte</strong><br>
                    <span style="color: {color}; font-size: 1.2em;">${levels.get('support', 0):.2f}</span><br>
                    <small>{distance_to_support:+.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
        
        with tech_cols[3]:
            atr = technical_analysis.get('atr')
            if atr:
                current_price = levels.get('current_price', 1)
                volatility_percent = (atr / current_price) * 100 if current_price > 0 else 0
                vol_status = "Alta" if volatility_percent > 3 else "Media" if volatility_percent > 1.5 else "Baja"
                st.markdown(f"""
                <div class="technical-indicator">
                    <strong>ATR (14)</strong><br>
                    <span style="font-size: 1.2em;">${atr:.2f}</span><br>
                    <small>Vol: {vol_status}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with tech_cols[4]:
            trend = technical_analysis.get('trend', {})
            if trend:
                overall_trend = "Alcista" if (trend.get('short_term') == 'bullish' and 
                                             trend.get('medium_term') == 'bullish') else "Bajista" if (
                                             trend.get('short_term') == 'bearish' and 
                                             trend.get('medium_term') == 'bearish') else "Mixta"
                trend_color = "#10b981" if overall_trend == "Alcista" else "#ef4444" if overall_trend == "Bajista" else "#6b7280"
                
                st.markdown(f"""
                <div class="technical-indicator">
                    <strong>Tendencia</strong><br>
                    <span style="color: {trend_color}; font-size: 1.2em;">{overall_trend}</span><br>
                    <small>Momentum: {trend.get('momentum', 'N/A').title()}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Señales de trading si están disponibles
        if 'signals' in technical_analysis:
            signals = technical_analysis['signals']
            
            st.markdown("#### 🎯 Señales de Trading")
            
            signal_color_map = {
                'strong_bullish': '#10b981',
                'bullish': '#34d399', 
                'neutral': '#6b7280',
                'bearish': '#f87171',
                'strong_bearish': '#ef4444'
            }
            
            overall_signal = signals.get('overall', 'neutral')
            signal_strength = signals.get('strength', 0)
            signal_color = signal_color_map.get(overall_signal, '#6b7280')
            
            signal_class = f"signal-{overall_signal.replace('_', '-')}" if '_' not in overall_signal else "signal-neutral"
            
            st.markdown(f"""
            <div class="sentiment-card {signal_class}">
                <h4>Señal General: <span style="color: {signal_color}">{overall_signal.replace('_', ' ').title()}</span></h4>
                <p><strong>Fuerza:</strong> {signal_strength}/100</p>
                <p><strong>Componentes:</strong></p>
                <ul>
            """, unsafe_allow_html=True)
            
            for component in signals.get('components', []):
                st.markdown(f"<li>{component}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error mostrando indicadores técnicos: {e}")

def display_sentiment_analysis(sentiment_data):
    """Muestra panel de análisis de sentiment"""
    if not sentiment_data or not utils_loaded:
        return
    
    st.markdown("### 📰 Análisis de Sentiment del Mercado")
    
    try:
        # Sentiment agregado
        aggregated = sentiment_data.get('aggregated', {})
        if aggregated:
            sentiment_cols = st.columns(3)
            
            with sentiment_cols[0]:
                sentiment_score = aggregated.get('score', 0)
                classification = aggregated.get('classification', 'neutral').replace('_', ' ').title()
                
                sentiment_color = '#10b981' if sentiment_score > 0.1 else '#ef4444' if sentiment_score < -0.1 else '#6b7280'
                
                st.markdown(f"""
                <div class="sentiment-card">
                    <h4>Sentiment General</h4>
                    <p style="color: {sentiment_color}; font-size: 1.5em;">{classification}</p>
                    <p><strong>Score:</strong> {sentiment_score:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with sentiment_cols[1]:
                confidence = aggregated.get('confidence', 'unknown').title()
                sources_count = aggregated.get('sources_count', 0)
                
                st.markdown(f"""
                <div class="sentiment-card">
                    <h4>Confiabilidad</h4>
                    <p style="font-size: 1.5em;">{confidence}</p>
                    <p><strong>Fuentes:</strong> {sources_count}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with sentiment_cols[2]:
                # Fear & Greed específico
                sources = sentiment_data.get('sources', {})
                if 'fear_greed' in sources:
                    fg = sources['fear_greed']
                    fg_value = fg.get('value', 0)
                    fg_class = fg.get('classification', 'Neutral')
                    
                    fg_color = '#ef4444' if fg_value < 40 else '#10b981' if fg_value > 60 else '#6b7280'
                    
                    st.markdown(f"""
                    <div class="sentiment-card">
                        <h4>Fear & Greed Index</h4>
                        <p style="color: {fg_color}; font-size: 1.5em;">{fg_value}/100</p>
                        <p><strong>{fg_class}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Noticias recientes si están disponibles
        news_summary = sentiment_data.get('news_summary', {})
        if news_summary and 'recent_headlines' in news_summary:
            st.markdown("#### 📰 Headlines Recientes")
            
            for headline in news_summary['recent_headlines'][:5]:
                sentiment = headline.get('sentiment', 'neutral')
                sentiment_emoji = "📈" if sentiment == 'positive' else "📉" if sentiment == 'negative' else "➡️"
                
                st.markdown(f"""
                <div class="news-headline">
                    {sentiment_emoji} <strong>{headline.get('title', '')[:100]}...</strong><br>
                    <small>Fuente: {headline.get('source', 'N/A')} | Sentiment: {sentiment.title()}</small>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error mostrando análisis de sentiment: {e}")

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

def display_realtime_ticker(symbol, market_type='spot'):
    """Muestra ticker de precio en tiempo real"""
    try:
        # Convertir símbolo si es necesario
        if market_type == 'spot' and '-USD' in symbol:
            binance_symbol = symbol.replace('-USD', 'USDT')
        else:
            binance_symbol = symbol
        
        # Obtener precio en tiempo real
        ticker_data = get_realtime_price(binance_symbol, market_type)
        
        if ticker_data:
            price = ticker_data.get('price', 0)
            change_24h = ticker_data.get('change_24h', 0)
            high_24h = ticker_data.get('high_24h', 0)
            low_24h = ticker_data.get('low_24h', 0)
            volume = ticker_data.get('volume_24h', 0)
            
            # Determinar color
            color = "#0ecb81" if change_24h >= 0 else "#f6465d"
            emoji = "📈" if change_24h >= 0 else "📉"
            
            # Formatear precio correctamente
            price_display = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
            high_display = f"${high_24h:,.2f}"
            low_display = f"${low_24h:,.2f}"
            
            # Mostrar ticker
            st.markdown(f"""
            <div style='background: #1a1a1a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid {color};'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <h2 style='margin: 0; color: #ffffff;'>{symbol.replace('-USD', '').replace('USDT', '')}/USDT</h2>
                        <p style='margin: 0; color: {color}; font-size: 2em; font-weight: bold;'>{price_display}</p>
                    </div>
                    <div style='text-align: right;'>
                        <p style='margin: 0; color: {color}; font-size: 1.5em;'>{emoji} {change_24h:+.2f}%</p>
                        <p style='margin: 0.5rem 0; color: #b0b0b0; font-size: 0.9em;'>24h Alto: {high_display}</p>
                        <p style='margin: 0; color: #b0b0b0; font-size: 0.9em;'>24h Bajo: {low_display}</p>
                    </div>
                    <div style='text-align: right;'>
                        <p style='margin: 0; color: #b0b0b0; font-size: 0.9em;'>Volumen 24h</p>
                        <p style='margin: 0; color: #ffffff; font-size: 1.2em;'>{volume:,.0f}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error mostrando ticker en tiempo real: {e}")

def main():
    """Función principal de la aplicación MEJORADA con auto-refresh"""
    
    # Inicializar estado
    initialize_session_state()
    setup_auto_refresh()
    
    # Validar configuración
    config_valid = validate_and_show_config()
    
    # Header principal
    app_title = Config.APP_TITLE if config_loaded else "Trading Assistant Pro"
    app_icon = Config.APP_ICON if config_loaded else "📈"
    
    try:
        ai_count = len(get_available_ais()) if config_loaded else 0
    except:
        ai_count = 0
    
    utils_status = "✅" if utils_loaded else "❌"
    
    st.markdown(f"""
    <div class="main-header">
        <h1>{app_icon} {app_title}</h1>
        <small>🤖 {ai_count} IA{'s' if ai_count != 1 else ''} • 📊 Utils: {utils_status}</small>
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
                    emoji = "₿" if metric['name'] == 'Bitcoin' else "⟠" if metric['name'] == 'Ethereum' else "◎" if metric['name'] == 'Solana' else "🔷"
                    price_display = f"${metric['price']:,.2f}" if metric['price'] >= 1 else f"${metric['price']:.4f}"
                    
                    st.metric(
                        f"{emoji} {metric['name']}", 
                        price_display,
                        f"{metric['change']:+.2f}%"
                    )
    else:
        st.warning("⚠️ No se pudieron cargar los precios en tiempo real.")
    
    # Layout principal: Gráfico y Chat
    col_chart, col_chat = st.columns([2.2, 1])
    
    with col_chart:
        # Mostrar ticker en tiempo real
        display_realtime_ticker(
            st.session_state.current_symbol,
            st.session_state.get('market_type', 'spot')
        )
        
        st.markdown(f"### 📊 Gráfico de {st.session_state.current_symbol.replace('-USD', '').replace('USDT', '')} - {st.session_state.current_timeframe.upper()}")
        
        # Obtener datos del gráfico
        with st.spinner("📈 Cargando datos del gráfico..."):
            chart_data = get_candlestick_data(
                st.session_state.current_symbol, 
                st.session_state.current_timeframe, 
                days=180,
                market_type=st.session_state.get("market_type", "spot")
            )
        
        # Obtener análisis técnico si está habilitado
        technical_analysis = None
        if st.session_state.show_technical_analysis and utils_loaded and not chart_data.empty:
            with st.spinner("🔧 Calculando indicadores técnicos..."):
                technical_analysis = get_technical_analysis(st.session_state.current_symbol, chart_data)
        
        if not chart_data.empty:
            fig = create_candlestick_chart(
                chart_data, 
                st.session_state.current_symbol.replace('-USD', '').replace('USDT', ''),
                st.session_state.current_timeframe,
                technical_analysis
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
                <small>💬 {len(st.session_state.chat_history)} mensajes • 🔧 Utils: {'✅' if utils_loaded else '❌'}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # CONTENEDOR DE CHAT CON ALTURA FIJA Y SCROLL
        chat_html = '<div class="chat-container">'
        
        # Generar HTML para cada mensaje
        for i, chat in enumerate(st.session_state.chat_history):
            if chat["role"] == "ai":
                if chat.get("provider") == "system":
                    chat_html += f"""
                    <div class="chat-message ai-message system-message">
                        <div class="message-content">{chat["message"]}</div>
                    </div>
                    """
                else:
                    provider_name = chat.get("provider", st.session_state.selected_ai or "IA")
                    ai_names = {
                        'claude': '🧠 Claude',
                        'openai': '💡 GPT-4',
                        'gemini': '⚡ Gemini'
                    }
                    provider_display = ai_names.get(provider_name, provider_name)
                    
                    chat_html += f"""
                    <div class="chat-message ai-message">
                        <div class="provider-info"><small><em>{provider_display}</em></small></div>
                        <div class="message-content">
                            {chat["message"]}
                        </div>
                    </div>
                    """
            elif chat["role"] == "user":
                chat_html += f"""
                <div class="chat-message user-message">
                    <div class="message-content">
                        {chat["message"]}
                    </div>
                </div>
            """
        
        chat_html += '</div>'
        
        # Mostrar el contenedor de chat
        st.markdown(chat_html, unsafe_allow_html=True)
        
        # Input del chat
        st.markdown("---")
        
        # Ejemplos de preguntas MEJORADOS
        with st.expander("💡 Ejemplos de consultas avanzadas"):
            if utils_loaded:
                example_queries = [
                    "Análisis técnico completo de Bitcoin",
                    "RSI y MACD actuales de Ethereum", 
                    "Sentiment del mercado crypto hoy",
                    "Divergencias en SOL timeframe 4h",
                    "Señales de Bollinger Bands",
                    "Análisis de volumen y momentum"
                ]
            else:
                example_queries = [
                    "Analiza Bitcoin en timeframe 4h",
                    "Puntos de entrada para Ethereum",
                    "Gestión de riesgo para SOL",
                    "Análisis del mercado crypto general"
                ]
            
            for example in example_queries:
                if st.button(f"📝 {example}", key=f"example_{example[:15]}", use_container_width=True):
                    if st.session_state.selected_ai and config_loaded:
                        st.session_state.chat_history.append({
                            "role": "user", 
                            "message": example
                        })
                        
                        sentiment_data = None
                        if st.session_state.show_sentiment_analysis and utils_loaded:
                            sentiment_data = get_sentiment_analysis(st.session_state.current_symbol)
                        
                        market_summary = create_market_summary_enhanced(
                            prices_data, fear_greed_data, market_data, 
                            technical_analysis, sentiment_data, 
                            st.session_state.current_symbol
                        )
                        
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
            placeholder="Ej: Análisis técnico completo de BTC, sentiment del mercado, señales RSI...",
            help="Pregunta sobre análisis técnico, sentiment, noticias, gestión de riesgo o estrategias",
            key="chat_input_field"
        )
        
        # Botones de acción del chat
        col_send, col_clear_msg = st.columns([2, 1])
        
        with col_send:
            if st.button("📤 Analizar", use_container_width=True, key="send_button") and user_input.strip():
                if not st.session_state.selected_ai or not config_loaded:
                    st.error("❌ No hay IAs disponibles. Configura al menos una API key.")
                else:
                    st.session_state.chat_history.append({
                        "role": "user", 
                        "message": user_input.strip() 
                    })
                    
                    sentiment_data = None
                    if st.session_state.show_sentiment_analysis and utils_loaded:
                        with st.spinner("📰 Analizando sentiment..."):
                            sentiment_data = get_sentiment_analysis(st.session_state.current_symbol)
                    
                    market_summary = create_market_summary_enhanced(
                        prices_data, fear_greed_data, market_data, 
                        technical_analysis, sentiment_data, 
                        st.session_state.current_symbol
                    )
                    
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
    
    # NUEVA SECCIÓN: Mostrar análisis técnico si está habilitado
    if st.session_state.show_technical_analysis and technical_analysis and utils_loaded:
        display_technical_indicators(technical_analysis, st.session_state.current_symbol.replace('-USD', '').replace('USDT', ''))
    
    # NUEVA SECCIÓN: Mostrar análisis de sentiment si está habilitado
    if st.session_state.show_sentiment_analysis and utils_loaded:
        with st.spinner("📰 Cargando análisis de sentiment..."):
            sentiment_data = get_sentiment_analysis(st.session_state.current_symbol)
        
        if sentiment_data:
            display_sentiment_analysis(sentiment_data)
    
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
                
                if fear_value <= 25:
                    delta_color = "inverse"
                elif fear_value >= 75:
                    delta_color = "normal"
                else:
                    delta_color = "off"
                
                st.metric(
                    "😨 Fear & Greed",
                    f"{fear_value}/100",
                    fear_classification,
                    delta_color=delta_color,
                    help="Índice de sentimiento del mercado: 0 = Extremo Miedo, 100 = Extrema Codicia"
                )
        except Exception as e:
            st.error(f"Error mostrando métricas globales: {e}")
    
    # Sección de análisis automático MEJORADA
    st.markdown("---")
    st.markdown("### 🎯 Análisis Rápido del Mercado")
    
    analysis_cols = st.columns(4)
    
    with analysis_cols[0]:
        if st.button("🚀 Análisis General", use_container_width=True):
            if st.session_state.selected_ai and config_loaded:
                sentiment_data = None
                if st.session_state.show_sentiment_analysis and utils_loaded:
                    sentiment_data = get_sentiment_analysis(st.session_state.current_symbol)
                
                market_summary = create_market_summary_enhanced(
                    prices_data, fear_greed_data, market_data, 
                    technical_analysis, sentiment_data, 
                    st.session_state.current_symbol
                )
                
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

{"6. Indicadores técnicos principales (RSI, MACD)" if technical_analysis else ""}
{"7. Sentiment de noticias y redes sociales" if sentiment_data else ""}

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
                sentiment_data = None
                if st.session_state.show_sentiment_analysis and utils_loaded:
                    sentiment_data = get_sentiment_analysis(st.session_state.current_symbol)
                
                market_summary = create_market_summary_enhanced(
                    prices_data, fear_greed_data, market_data, 
                    technical_analysis, sentiment_data, 
                    st.session_state.current_symbol
                )
                
                risk_prompt = f"""Como especialista en gestión de riesgo para crypto trading, necesito recomendaciones específicas.

CONTEXTO ACTUAL:
- Volatilidad del mercado: {"Alta" if fear_greed_data and int(fear_greed_data['value']) < 40 else "Media"}
- Bitcoin dominancia: {market_data['market_cap_percentage']['btc']:.1f}% si market_data else 'N/A'
- Sentiment general: {fear_greed_data['value_classification'] if fear_greed_data else 'N/A'}
{"- RSI actual: " + str(round(technical_analysis.get('rsi', 0), 1)) if technical_analysis and technical_analysis.get('rsi') else ""}
{"- ATR (volatilidad): $" + str(round(technical_analysis.get('atr', 0), 2)) if technical_analysis and technical_analysis.get('atr') else ""}

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
                sentiment_data = None
                if st.session_state.show_sentiment_analysis and utils_loaded:
                    sentiment_data = get_sentiment_analysis(st.session_state.current_symbol)
                
                market_summary = create_market_summary_enhanced(
                    prices_data, fear_greed_data, market_data, 
                    technical_analysis, sentiment_data, 
                    st.session_state.current_symbol
                )
                
                technical_info = ""
                if technical_analysis:
                    levels = technical_analysis.get('levels', {})
                    if levels:
                        technical_info = f"""
DATOS TÉCNICOS ACTUALES:
- Precio actual: ${levels.get('current_price', 0):,.2f}
- Resistencia: ${levels.get('resistance', 0):,.2f} ({levels.get('distance_to_resistance', 0):+.1f}%)
- Soporte: ${levels.get('support', 0):,.2f} ({levels.get('distance_to_support', 0):+.1f}%)
- RSI: {technical_analysis.get('rsi', 'N/A')}
- ATR: ${technical_analysis.get('atr', 0):.2f}
- Señal general: {technical_analysis.get('signals', {}).get('overall', 'neutral')}
"""

                entry_prompt = f"""Como trader técnico especializado, identifica puntos de entrada óptimos.

PRECIOS ACTUALES:
- BTC: ${prices_data['bitcoin']['usd']:,.2f} ({prices_data['bitcoin']['usd_24h_change']:+.2f}%)
- ETH: ${prices_data['ethereum']['usd']:,.2f} ({prices_data['ethereum']['usd_24h_change']:+.2f}%)
- SOL: ${prices_data['solana']['usd']:.2f} ({prices_data['solana']['usd_24h_change']:+.2f}%)

ASSET PRINCIPAL: {st.session_state.current_symbol.replace('-USD', '').replace('USDT', '')}
TIMEFRAME: {st.session_state.current_timeframe}

{technical_info}

ANÁLISIS REQUERIDO:
1. Niveles de entrada específicos para {st.session_state.current_symbol.replace('-USD', '').replace('USDT', '')}
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
                            "message": f"🎯 Puntos de Entrada para {st.session_state.current_symbol.replace('-USD', '').replace('USDT', '')}"
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
    
    with analysis_cols[3]:
        if st.button("📰 Análisis Sentiment", use_container_width=True):
            if st.session_state.selected_ai and config_loaded:
                sentiment_data = None
                if utils_loaded:
                    with st.spinner("📰 Obteniendo noticias..."):
                        sentiment_data = get_sentiment_analysis(st.session_state.current_symbol)
                
                market_summary = create_market_summary_enhanced(
                    prices_data, fear_greed_data, market_data, 
                    technical_analysis, sentiment_data, 
                    st.session_state.current_symbol
                )
                
                sentiment_info = ""
                if sentiment_data:
                    aggregated = sentiment_data.get('aggregated', {})
                    sources = sentiment_data.get('sources', {})
                    sentiment_info = f"""
DATOS DE SENTIMENT ACTUALES:
- Sentiment general: {aggregated.get('classification', 'neutral').replace('_', ' ').title()}
- Score agregado: {aggregated.get('score', 0):.2f}
- Confianza: {aggregated.get('confidence', 'unknown').title()}
- Fear & Greed: {sources.get('fear_greed', {}).get('value', 'N/A')}/100
- Fuentes analizadas: {aggregated.get('sources_count', 0)}
"""

                sentiment_prompt = f"""Como especialista en análisis de sentiment para crypto, dame un análisis completo del sentimiento actual del mercado.

{sentiment_info}

CONTEXTO GENERAL:
- Bitcoin: ${prices_data['bitcoin']['usd']:,.2f} ({prices_data['bitcoin']['usd_24h_change']:+.2f}%)
- Market Cap: ${market_data['total_market_cap']['usd']/1e12:.2f}T
- Dominancia BTC: {market_data['market_cap_percentage']['btc']:.1f}%

ANÁLISIS REQUERIDO:
1. Interpretación del sentiment actual y su impacto
2. Comparación histórica - ¿estamos en extremos?
3. Correlación entre sentiment y movimientos de precio
4. Señales contrarias vs seguimiento de tendencia
5. Noticias y eventos que influyen el sentiment
6. Recomendaciones según el sentiment actual
7. Timing de mercado basado en psicología

{"8. Integración con señales técnicas" if technical_analysis else ""}

Enfócate en cómo usar el sentiment para tomar mejores decisiones de trading. Respuesta práctica y actionable."""
                
                with st.spinner("📰 Analizando sentiment del mercado..."):
                    try:
                        analysis = analyze_market(
                            sentiment_prompt,
                            market_summary,
                            provider=st.session_state.selected_ai
                        )
                        
                        st.session_state.chat_history.append({
                            "role": "user",
                            "message": "📰 Análisis de Sentiment del Mercado"
                        })
                        
                        st.session_state.chat_history.append({
                            "role": "ai",
                            "message": analysis,
                            "provider": st.session_state.selected_ai
                        })
                    except Exception as e:
                        st.error(f"Error en análisis de sentiment: {e}")
                        
                st.rerun()
            else:
                st.error("❌ No hay IAs disponibles")
    
    # IMPLEMENTAR AUTO-REFRESH
    if st.session_state.get('auto_refresh_enabled', False):
        import time
        
        status_placeholder = st.empty()
        refresh_interval = st.session_state.get('refresh_interval', 5)
        
        # Contador regresivo
        for remaining in range(refresh_interval, 0, -1):
            status_placeholder.info(f"🔄 Próxima actualización en {remaining} segundos...")
            time.sleep(1)
        
        status_placeholder.empty()
        st.rerun()
        
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #64748b; padding: 1.5rem;'>
        <strong>📈 Trading Assistant Pro v2.2</strong> • Estilo Binance<br>
        🤖 Powered by Claude, GPT-4 & Gemini • 📊 Todos los pares Binance Spot/Futures<br>
        {"✅" if utils_loaded else "❌"} <small>Indicadores técnicos • Análisis de sentiment • Gestión de riesgo</small><br>
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
                        "Market Type": st.session_state.get('market_type', 'spot'),
                        "Selected AI": st.session_state.selected_ai,
                        "Chat History Length": len(st.session_state.chat_history),
                        "Show AI Comparison": st.session_state.show_ai_comparison,
                        "Technical Analysis Enabled": st.session_state.show_technical_analysis,
                        "Sentiment Analysis Enabled": st.session_state.show_sentiment_analysis,
                        "Auto Refresh Enabled": st.session_state.get('auto_refresh_enabled', False),
                        "Refresh Interval": st.session_state.get('refresh_interval', 5)
                    },
                    "Market Data Status": {
                        "Prices Data": bool(prices_data),
                        "Market Data": bool(market_data),
                        "Fear & Greed Data": bool(fear_greed_data),
                        "Chart Data": not chart_data.empty if 'chart_data' in locals() else False,
                        "Technical Analysis": bool(technical_analysis)
                    },
                    "System Status": {
                        "Config Loaded": config_loaded,
                        "Utils Loaded": utils_loaded,
                        "AI Status": get_ai_stats() if config_loaded else "Not loaded",
                        "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                    }
                }
                st.json(debug_info)
            except Exception as e:
                st.error(f"Error en debug info: {e}")
    

# Funciones adicionales para compatibilidad
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
        "Streamlit Version": st.__version__ if hasattr(st, '__version__') else "Unknown",
        "Config Loaded": config_loaded,
        "Utils Loaded": utils_loaded
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
        print(f"📊 Utils loaded: {utils_loaded}")
        
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
               - utils/ (carpeta completa)  
               - .env (con tus API keys)
            """)

# Meta información del archivo actualizada
__version__ = "2.1.0"
__author__ = "Trading Assistant Pro"
__description__ = "Aplicación de análisis de trading con IA múltiple + Utils integrados compatible con Python 3.8+"
__python_requires__ = ">=3.8"
__features__ = [
    "Multiple AI providers (Claude, GPT-4, Gemini)",
    "Advanced technical analysis (RSI, MACD, Bollinger Bands)",
    "Sentiment analysis (News + Fear & Greed)",
    "Real-time market data",
    "Professional trading charts",
    "Risk management tools"
]

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
    },
    "features": {
        "technical_analysis": True,
        "sentiment_analysis": True,
        "ai_providers": True,
        "real_time_data": True
    }
}