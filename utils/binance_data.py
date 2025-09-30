"""
Binance Data Handler - Obtiene todos los símbolos y datos de Spot y Futures
"""
import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from functools import lru_cache
import time
import json
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

class BinanceDataHandler:
    """Maneja todos los datos de Binance Spot y Futures"""
    
    def __init__(self):
        self.spot_base_url = "https://api.binance.com/api/v3"
        self.futures_base_url = "https://fapi.binance.com/fapi/v1"
        self._symbols_cache = {}
        self._cache_time = {}
        self.cache_duration = 300  # 5 minutos
    
    def _is_cache_valid(self, key: str) -> bool:
        """Verifica si el cache es válido"""
        if key not in self._cache_time:
            return False
        return (time.time() - self._cache_time[key]) < self.cache_duration
    
    def get_all_spot_symbols(self) -> List[Dict]:
        """Obtiene TODOS los símbolos de Spot trading"""
        cache_key = 'spot_symbols'
        
        if self._is_cache_valid(cache_key):
            return self._symbols_cache[cache_key]
        
        try:
            response = requests.get(f"{self.spot_base_url}/exchangeInfo", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Filtrar solo pares USDT activos
            symbols = []
            for symbol_info in data['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['quoteAsset'] == 'USDT' and
                    symbol_info['isSpotTradingAllowed']):
                    
                    symbols.append({
                        'symbol': symbol_info['symbol'],
                        'baseAsset': symbol_info['baseAsset'],
                        'quoteAsset': symbol_info['quoteAsset'],
                        'status': symbol_info['status'],
                        'type': 'SPOT'
                    })
            
            # Ordenar alfabéticamente
            symbols.sort(key=lambda x: x['symbol'])
            
            # Actualizar cache
            self._symbols_cache[cache_key] = symbols
            self._cache_time[cache_key] = time.time()
            
            logger.info(f"Cargados {len(symbols)} símbolos SPOT de Binance")
            return symbols
            
        except Exception as e:
            logger.error(f"Error obteniendo símbolos SPOT: {e}")
            return []
    
    def get_all_futures_symbols(self) -> List[Dict]:
        """Obtiene TODOS los símbolos de Futures trading"""
        cache_key = 'futures_symbols'
        
        if self._is_cache_valid(cache_key):
            return self._symbols_cache[cache_key]
        
        try:
            response = requests.get(f"{self.futures_base_url}/exchangeInfo", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Filtrar solo contratos USDT perpetuos activos
            symbols = []
            for symbol_info in data['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['quoteAsset'] == 'USDT' and
                    symbol_info['contractType'] == 'PERPETUAL'):
                    
                    symbols.append({
                        'symbol': symbol_info['symbol'],
                        'baseAsset': symbol_info['baseAsset'],
                        'quoteAsset': symbol_info['quoteAsset'],
                        'status': symbol_info['status'],
                        'type': 'FUTURES',
                        'contractType': symbol_info['contractType']
                    })
            
            # Ordenar alfabéticamente
            symbols.sort(key=lambda x: x['symbol'])
            
            # Actualizar cache
            self._symbols_cache[cache_key] = symbols
            self._cache_time[cache_key] = time.time()
            
            logger.info(f"Cargados {len(symbols)} símbolos FUTURES de Binance")
            return symbols
            
        except Exception as e:
            logger.error(f"Error obteniendo símbolos FUTURES: {e}")
            return []
    
    def get_24h_ticker_data(self, market_type: str = 'spot') -> Dict[str, Dict]:
        """Obtiene datos de precio de 24h para todos los símbolos"""
        try:
            if market_type == 'spot':
                url = f"{self.spot_base_url}/ticker/24hr"
            else:
                url = f"{self.futures_base_url}/ticker/24hr"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Crear diccionario para acceso rápido
            ticker_dict = {}
            for ticker in data:
                symbol = ticker['symbol']
                if ticker.get('quoteAsset') == 'USDT' or symbol.endswith('USDT'):
                    ticker_dict[symbol] = {
                        'price': float(ticker['lastPrice']),
                        'change_24h': float(ticker['priceChangePercent']),
                        'high_24h': float(ticker['highPrice']),
                        'low_24h': float(ticker['lowPrice']),
                        'volume_24h': float(ticker['volume']),
                        'quote_volume': float(ticker['quoteVolume'])
                    }
            
            return ticker_dict
            
        except Exception as e:
            logger.error(f"Error obteniendo ticker data: {e}")
            return {}
    
    def get_top_gainers_losers(self, market_type: str = 'spot', limit: int = 10) -> Tuple[List, List]:
        """Obtiene los mayores ganadores y perdedores"""
        ticker_data = self.get_24h_ticker_data(market_type)
        
        if not ticker_data:
            return [], []
        
        # Convertir a lista y ordenar
        tickers_list = [
            {'symbol': symbol, **data} 
            for symbol, data in ticker_data.items()
        ]
        
        # Ordenar por cambio porcentual
        gainers = sorted(tickers_list, key=lambda x: x['change_24h'], reverse=True)[:limit]
        losers = sorted(tickers_list, key=lambda x: x['change_24h'])[:limit]
        
        return gainers, losers
    
    def get_klines(self, symbol: str, interval: str, limit: int = 1000, market_type: str = 'spot') -> pd.DataFrame:
        """Obtiene datos de velas (klines) de Binance"""
        try:
            if market_type == 'spot':
                url = f"{self.spot_base_url}/klines"
            else:
                url = f"{self.futures_base_url}/klines"
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Convertir a DataFrame
            df = pd.DataFrame(data, columns=[
                'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
                'CloseTime', 'QuoteVolume', 'Trades', 
                'TakerBuyBase', 'TakerBuyQuote', 'Ignore'
            ])
            
            # Convertir tipos
            df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
            df.set_index('OpenTime', inplace=True)
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            logger.error(f"Error obteniendo klines para {symbol}: {e}")
            return pd.DataFrame()
    
    def search_symbols(self, query: str, market_type: str = 'spot') -> List[Dict]:
        """Busca símbolos que coincidan con el query"""
        if market_type == 'spot':
            all_symbols = self.get_all_spot_symbols()
        else:
            all_symbols = self.get_all_futures_symbols()
        
        query_upper = query.upper()
        
        # Buscar coincidencias
        matches = [
            sym for sym in all_symbols
            if query_upper in sym['symbol'] or query_upper in sym['baseAsset']
        ]
        
        return matches[:50]  # Limitar a 50 resultados
    
    def get_symbol_info(self, symbol: str, market_type: str = 'spot') -> Optional[Dict]:
        """Obtiene información detallada de un símbolo"""
        if market_type == 'spot':
            all_symbols = self.get_all_spot_symbols()
        else:
            all_symbols = self.get_all_futures_symbols()
        
        for sym in all_symbols:
            if sym['symbol'] == symbol:
                return sym
        
        return None


class BinanceWebSocket:
    """Maneja conexión WebSocket de Binance para datos en tiempo real"""
    
    def __init__(self):
        self.ws = None
        self.is_running = False
        self.last_price = {}
        self.last_update = {}
    
    def get_stream_url(self, symbol: str, market_type: str = 'spot') -> str:
        """Genera URL del stream de WebSocket"""
        symbol_lower = symbol.lower()
        
        if market_type == 'spot':
            return f"wss://stream.binance.com:9443/ws/{symbol_lower}@ticker"
        else:
            return f"wss://fstream.binance.com/ws/{symbol_lower}@ticker"
    
    def get_kline_stream_url(self, symbol: str, interval: str, market_type: str = 'spot') -> str:
        """Genera URL del stream de klines (velas)"""
        symbol_lower = symbol.lower()
        
        if market_type == 'spot':
            return f"wss://stream.binance.com:9443/ws/{symbol_lower}@kline_{interval}"
        else:
            return f"wss://fstream.binance.com/ws/{symbol_lower}@kline_{interval}"
    
    def parse_ticker_message(self, message: dict) -> dict:
        """Parsea mensaje de ticker del WebSocket"""
        try:
            return {
                'symbol': message.get('s', ''),
                'price': float(message.get('c', 0)),
                'change_24h': float(message.get('P', 0)),
                'high_24h': float(message.get('h', 0)),
                'low_24h': float(message.get('l', 0)),
                'volume_24h': float(message.get('v', 0)),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error parseando ticker: {e}")
            return {}
    
    def parse_kline_message(self, message: dict) -> dict:
        """Parsea mensaje de kline del WebSocket"""
        try:
            kline = message.get('k', {})
            return {
                'time': pd.to_datetime(kline.get('t'), unit='ms'),
                'open': float(kline.get('o', 0)),
                'high': float(kline.get('h', 0)),
                'low': float(kline.get('l', 0)),
                'close': float(kline.get('c', 0)),
                'volume': float(kline.get('v', 0)),
                'is_closed': kline.get('x', False)  # True si la vela está cerrada
            }
        except Exception as e:
            logger.error(f"Error parseando kline: {e}")
            return {}


def get_realtime_price(symbol: str, market_type: str = 'spot') -> dict:
    """Obtiene precio en tiempo real desde API REST"""
    try:
        # Convertir símbolo si viene en formato BTC-USD para Futures
        if market_type == 'futures' and '-USD' in symbol:
            symbol = symbol.replace('-USD', 'USDT')
        
        if market_type == 'spot':
            url = "https://api.binance.com/api/v3/ticker/24hr"
            params = {'symbol': symbol}
        else:
            url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
            params = {'symbol': symbol}
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        return {
            'price': float(data.get('lastPrice', 0)),
            'change_24h': float(data.get('priceChangePercent', 0)),
            'high_24h': float(data.get('highPrice', 0)),
            'low_24h': float(data.get('lowPrice', 0)),
            'volume_24h': float(data.get('volume', 0)),
            'timestamp': datetime.now()
        }
    except Exception as e:
        logger.error(f"Error obteniendo precio en tiempo real: {e}")
        return {}


# Instancias globales
binance_handler = BinanceDataHandler()
binance_ws = BinanceWebSocket()

# Funciones de conveniencia
def get_all_symbols(market_type: str = 'spot') -> List[Dict]:
    """Obtiene todos los símbolos disponibles"""
    if market_type == 'spot':
        return binance_handler.get_all_spot_symbols()
    else:
        return binance_handler.get_all_futures_symbols()

def get_ticker_data(market_type: str = 'spot') -> Dict:
    """Obtiene datos de ticker de 24h"""
    return binance_handler.get_24h_ticker_data(market_type)

def get_top_movers(market_type: str = 'spot', limit: int = 10) -> Tuple[List, List]:
    """Obtiene ganadores y perdedores"""
    return binance_handler.get_top_gainers_losers(market_type, limit)

def search_symbol(query: str, market_type: str = 'spot') -> List[Dict]:
    """Busca símbolos"""
    return binance_handler.search_symbols(query, market_type)

def get_chart_data(symbol: str, interval: str, limit: int = 1000, market_type: str = 'spot') -> pd.DataFrame:
    """Obtiene datos del gráfico"""
    return binance_handler.get_klines(symbol, interval, limit, market_type)