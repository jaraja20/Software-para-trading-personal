"""
Indicadores técnicos avanzados para Trading Assistant Pro
Requiere: pip install pandas numpy talib
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Clase para calcular indicadores técnicos avanzados"""
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        """MACD indicator"""
        exp1 = data.ewm(span=fast).mean()
        exp2 = data.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> dict:
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        return {
            'middle': sma,
            'upper': sma + (std * std_dev),
            'lower': sma - (std * std_dev),
            'bandwidth': ((sma + (std * std_dev)) - (sma - (std * std_dev))) / sma * 100
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> dict:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> dict:
        """Fibonacci retracement levels"""
        diff = high - low
        return {
            'level_0': high,
            'level_236': high - 0.236 * diff,
            'level_382': high - 0.382 * diff,
            'level_500': high - 0.500 * diff,
            'level_618': high - 0.618 * diff,
            'level_786': high - 0.786 * diff,
            'level_100': low
        }
    
    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume Simple Moving Average"""
        return volume.rolling(window=period).mean()
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    @staticmethod
    def detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 20) -> dict:
        """Detect bullish/bearish divergences"""
        price_peaks = []
        indicator_peaks = []
        
        # Find peaks in price and indicator
        for i in range(lookback, len(price) - lookback):
            if price.iloc[i] == price.iloc[i-lookback:i+lookback+1].max():
                price_peaks.append((i, price.iloc[i]))
            if indicator.iloc[i] == indicator.iloc[i-lookback:i+lookback+1].max():
                indicator_peaks.append((i, indicator.iloc[i]))
        
        divergences = []
        
        # Check for divergences between recent peaks
        if len(price_peaks) >= 2 and len(indicator_peaks) >= 2:
            recent_price_peaks = price_peaks[-2:]
            recent_indicator_peaks = indicator_peaks[-2:]
            
            # Bullish divergence: price makes lower low, indicator makes higher low
            if (recent_price_peaks[1][1] < recent_price_peaks[0][1] and 
                recent_indicator_peaks[1][1] > recent_indicator_peaks[0][1]):
                divergences.append('bullish')
            
            # Bearish divergence: price makes higher high, indicator makes lower high
            if (recent_price_peaks[1][1] > recent_price_peaks[0][1] and 
                recent_indicator_peaks[1][1] < recent_indicator_peaks[0][1]):
                divergences.append('bearish')
        
        return {
            'divergences': divergences,
            'price_peaks': price_peaks,
            'indicator_peaks': indicator_peaks
        }

class MarketAnalyzer:
    """Análisis avanzado del mercado"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def comprehensive_analysis(self, data: pd.DataFrame) -> dict:
        """Análisis técnico completo"""
        if data.empty or len(data) < 50:
            return {"error": "Insufficient data for analysis"}
        
        try:
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            # Calcular todos los indicadores
            analysis = {
                'rsi': self.indicators.rsi(close).iloc[-1] if len(close) >= 14 else None,
                'macd': self.indicators.macd(close),
                'bollinger': self.indicators.bollinger_bands(close),
                'stochastic': self.indicators.stochastic(high, low, close),
                'atr': self.indicators.atr(high, low, close).iloc[-1] if len(close) >= 14 else None,
                'williams_r': self.indicators.williams_r(high, low, close).iloc[-1] if len(close) >= 14 else None,
                'volume_sma': self.indicators.volume_sma(volume).iloc[-1] if len(volume) >= 20 else None,
                'fibonacci': self.indicators.fibonacci_retracement(high.max(), low.min())
            }
            
            # Detectar divergencias
            if len(close) >= 40:
                rsi_series = self.indicators.rsi(close)
                divergence_analysis = self.indicators.detect_divergence(close, rsi_series)
                analysis['divergences'] = divergence_analysis
            
            # Análisis de tendencia
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            
            current_price = close.iloc[-1]
            trend_analysis = {
                'short_term': 'bullish' if current_price > sma_20.iloc[-1] else 'bearish',
                'medium_term': 'bullish' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'bearish',
                'momentum': 'strong' if abs(close.pct_change().iloc[-1]) > 0.03 else 'weak'
            }
            analysis['trend'] = trend_analysis
            
            # Niveles de soporte y resistencia
            recent_data = close.tail(50)
            resistance_level = recent_data.quantile(0.9)
            support_level = recent_data.quantile(0.1)
            
            analysis['levels'] = {
                'current_price': current_price,
                'resistance': resistance_level,
                'support': support_level,
                'distance_to_resistance': (resistance_level - current_price) / current_price * 100,
                'distance_to_support': (current_price - support_level) / current_price * 100
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive_analysis: {e}")
            return {"error": str(e)}
    
    def generate_signals(self, analysis: dict) -> dict:
        """Generar señales de trading basadas en el análisis"""
        signals = {
            'overall': 'neutral',
            'strength': 0,  # -100 to 100
            'components': []
        }
        
        try:
            strength_score = 0
            signal_count = 0
            
            # RSI signals
            if analysis.get('rsi'):
                rsi = analysis['rsi']
                if rsi < 30:
                    signals['components'].append('RSI oversold - bullish')
                    strength_score += 20
                elif rsi > 70:
                    signals['components'].append('RSI overbought - bearish')
                    strength_score -= 20
                signal_count += 1
            
            # MACD signals
            macd_data = analysis.get('macd', {})
            if 'macd' in macd_data and 'signal' in macd_data:
                macd_line = macd_data['macd'].iloc[-1]
                signal_line = macd_data['signal'].iloc[-1]
                
                if macd_line > signal_line:
                    signals['components'].append('MACD bullish crossover')
                    strength_score += 15
                else:
                    signals['components'].append('MACD bearish crossover')
                    strength_score -= 15
                signal_count += 1
            
            # Bollinger Bands signals
            bollinger = analysis.get('bollinger', {})
            levels = analysis.get('levels', {})
            if 'lower' in bollinger and 'upper' in bollinger and 'current_price' in levels:
                current_price = levels['current_price']
                lower_band = bollinger['lower'].iloc[-1]
                upper_band = bollinger['upper'].iloc[-1]
                
                if current_price <= lower_band:
                    signals['components'].append('Price at lower Bollinger Band - potential bounce')
                    strength_score += 25
                elif current_price >= upper_band:
                    signals['components'].append('Price at upper Bollinger Band - potential pullback')
                    strength_score -= 25
                signal_count += 1
            
            # Trend signals
            trend = analysis.get('trend', {})
            if trend.get('short_term') == 'bullish' and trend.get('medium_term') == 'bullish':
                signals['components'].append('Strong uptrend across timeframes')
                strength_score += 30
            elif trend.get('short_term') == 'bearish' and trend.get('medium_term') == 'bearish':
                signals['components'].append('Strong downtrend across timeframes')
                strength_score -= 30
            
            # Divergence signals
            divergences = analysis.get('divergences', {}).get('divergences', [])
            for div in divergences:
                if div == 'bullish':
                    signals['components'].append('Bullish divergence detected - potential reversal')
                    strength_score += 35
                elif div == 'bearish':
                    signals['components'].append('Bearish divergence detected - potential reversal')
                    strength_score -= 35
            
            # Calculate final score
            if signal_count > 0:
                signals['strength'] = max(-100, min(100, strength_score))
            
            # Determine overall signal
            if signals['strength'] > 30:
                signals['overall'] = 'strong_bullish'
            elif signals['strength'] > 10:
                signals['overall'] = 'bullish'
            elif signals['strength'] < -30:
                signals['overall'] = 'strong_bearish'
            elif signals['strength'] < -10:
                signals['overall'] = 'bearish'
            else:
                signals['overall'] = 'neutral'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return signals

def format_analysis_for_ai(analysis: dict, symbol: str) -> str:
    """Formatea el análisis técnico para ser usado por la IA"""
    if 'error' in analysis:
        return f"Error en análisis técnico de {symbol}: {analysis['error']}"
    
    try:
        output = f"ANÁLISIS TÉCNICO DETALLADO - {symbol}:\n\n"
        
        # Información básica
        levels = analysis.get('levels', {})
        if levels:
            output += f"Precio actual: ${levels.get('current_price', 0):.2f}\n"
            output += f"Resistencia: ${levels.get('resistance', 0):.2f} ({levels.get('distance_to_resistance', 0):+.1f}%)\n"
            output += f"Soporte: ${levels.get('support', 0):.2f} ({levels.get('distance_to_support', 0):+.1f}%)\n\n"
        
        # RSI
        rsi = analysis.get('rsi')
        if rsi:
            rsi_status = "Sobreventa" if rsi < 30 else "Sobrecompra" if rsi > 70 else "Neutral"
            output += f"RSI (14): {rsi:.1f} - {rsi_status}\n"
        
        # MACD
        macd_data = analysis.get('macd', {})
        if 'macd' in macd_data and 'signal' in macd_data:
            macd_current = macd_data['macd'].iloc[-1]
            signal_current = macd_data['signal'].iloc[-1]
            macd_trend = "Alcista" if macd_current > signal_current else "Bajista"
            output += f"MACD: {macd_trend} (MACD: {macd_current:.2f}, Signal: {signal_current:.2f})\n"
        
        # ATR (Volatilidad)
        atr = analysis.get('atr')
        if atr:
            output += f"ATR (14): {atr:.2f} - Volatilidad promedio\n"
        
        # Tendencia
        trend = analysis.get('trend', {})
        if trend:
            output += f"\nTENDENCIA:\n"
            output += f"Corto plazo: {trend.get('short_term', 'unknown').title()}\n"
            output += f"Mediano plazo: {trend.get('medium_term', 'unknown').title()}\n"
            output += f"Momentum: {trend.get('momentum', 'unknown').title()}\n"
        
        # Fibonacci levels
        fib = analysis.get('fibonacci', {})
        if fib:
            output += f"\nNIVELES FIBONACCI:\n"
            output += f"0.382: ${fib.get('level_382', 0):.2f}\n"
            output += f"0.500: ${fib.get('level_500', 0):.2f}\n"
            output += f"0.618: ${fib.get('level_618', 0):.2f}\n"
        
        # Divergencias
        divergences = analysis.get('divergences', {}).get('divergences', [])
        if divergences:
            output += f"\nDIVERGENCIAS DETECTADAS:\n"
            for div in divergences:
                output += f"- {div.title()} divergence\n"
        
        return output
        
    except Exception as e:
        logger.error(f"Error formatting analysis: {e}")
        return f"Error formateando análisis de {symbol}"