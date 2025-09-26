"""
News and Sentiment Analysis para Trading Assistant Pro
APIs: NewsAPI, CryptoPanic, Reddit
"""
import requests
from functools import lru_cache
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any
import re
from collections import Counter

logger = logging.getLogger(__name__)

class NewsAnalyzer:
    """Analizador de noticias financieras"""
    
    def __init__(self, news_api_key: str = None):
        import os
        self.news_api_key = news_api_key or os.getenv('NEWS_API_KEY', '2208cef67ba24d11a2d48ef4789241d1')
        self.cryptopanic_base = "https://cryptopanic.com/api/v1/posts/"
        self.newsapi_base = "https://newsapi.org/v2/everything"
        
    def get_crypto_news(self, symbols: List[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Obtener noticias de crypto desde múltiples fuentes"""
        if not symbols:
            symbols = ['bitcoin', 'ethereum', 'solana']
        
        all_news = []
        
        # CryptoPanic (gratis, no requiere API key)
        crypto_panic_news = self._get_cryptopanic_news(hours)
        all_news.extend(crypto_panic_news)
        
        # NewsAPI (requiere key)
        if self.news_api_key:
            newsapi_news = self._get_newsapi_crypto(symbols, hours)
            all_news.extend(newsapi_news)
        
        # Reddit headlines (usando RSS - gratis)
        reddit_news = self._get_reddit_crypto_news()
        all_news.extend(reddit_news)
        
        # Analizar sentiment
        analyzed_news = self._analyze_news_sentiment(all_news)
        
        return {
            'news': analyzed_news,
            'summary': self._generate_news_summary(analyzed_news),
            'sentiment_score': self._calculate_overall_sentiment(analyzed_news)
        }
    
    @lru_cache(maxsize=32)
    def _get_cryptopanic_news(self, hours: int) -> List[Dict]:
        """Obtener noticias de CryptoPanic (gratis)"""
        try:
            # CryptoPanic tiene endpoints públicos limitados
            url = f"{self.cryptopanic_base}?filter=hot"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results') if data else None
                if not results:
                    return []

                news = []
                for item in results[:20]:  # Limitar a 20
                    news.append({
                        'title': item.get('title', ''),
                        'url': item.get('url', ''),
                        'published_at': item.get('published_at', ''),
                        'source': 'CryptoPanic',
                        'currencies': [c.get('code', '') for c in item.get('currencies', [])]
                    })
                
                return news
            elif response.status_code == 429:
                logger.warning("⚠️ Rate limit alcanzado en CryptoPanic, devolviendo fallback.")
                return [{
                    "title": "⚠️ Demasiadas peticiones a CryptoPanic",
                    "url": "",
                    "published_at": datetime.now().isoformat(),
                    "source": "System",
                    "currencies": []
                }]  

            else:
                logger.error(f"CryptoPanic API error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching CryptoPanic news: {e}")
            return []
    
    def _get_newsapi_crypto(self, symbols: List[str], hours: int) -> List[Dict]:
        """Obtener noticias de NewsAPI"""
        if not self.news_api_key:
            return []
        
        try:
            from_date = (datetime.now() - timedelta(hours=hours)).isoformat()
            query = ' OR '.join([f'"{symbol}"' for symbol in symbols])
            
            params = {
                'q': f'{query} AND (cryptocurrency OR crypto OR blockchain)',
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': self.news_api_key,
                'pageSize': 30
            }
            
            response = requests.get(self.newsapi_base, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                news = []
                
                for article in data.get('articles', []):
                    news.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'currencies': self._extract_currencies_from_text(article.get('title', '') + ' ' + article.get('description', ''))
                    })
                
                return news
        except Exception as e:
            logger.error(f"Error fetching NewsAPI: {e}")
            return []
    
    def _get_reddit_crypto_news(self) -> List[Dict]:
        """Obtener headlines de Reddit crypto subreddits"""
        try:
            # Reddit RSS feeds (públicos)
            subreddits = ['cryptocurrency', 'bitcoin', 'ethereum']
            news = []
            
            for subreddit in subreddits:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=10"
                headers = {'User-Agent': 'TradingBot/1.0'}
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for post in data.get('data', {}).get('children', []):
                        post_data = post.get('data', {})
                        
                        news.append({
                            'title': post_data.get('title', ''),
                            'url': f"https://reddit.com{post_data.get('permalink', '')}",
                            'published_at': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat(),
                            'source': f'r/{subreddit}',
                            'score': post_data.get('score', 0),
                            'currencies': self._extract_currencies_from_text(post_data.get('title', ''))
                        })
            
            return news
            
        except Exception as e:
            logger.error(f"Error fetching Reddit news: {e}")
            return []
    
    def _extract_currencies_from_text(self, text: str) -> List[str]:
        """Extraer menciones de criptomonedas del texto"""
        text = text.lower()
        currencies = []
        
        # Mapeo de términos a símbolos
        currency_map = {
            'bitcoin': 'BTC', 'btc': 'BTC',
            'ethereum': 'ETH', 'eth': 'ETH', 'ether': 'ETH',
            'solana': 'SOL', 'sol': 'SOL',
            'cardano': 'ADA', 'ada': 'ADA',
            'polygon': 'MATIC', 'matic': 'MATIC',
            'avalanche': 'AVAX', 'avax': 'AVAX'
        }
        
        for term, symbol in currency_map.items():
            if term in text and symbol not in currencies:
                currencies.append(symbol)
        
        return currencies
    
    def _analyze_news_sentiment(self, news: List[Dict]) -> List[Dict]:
        """Analizar sentiment de las noticias usando keywords"""
        positive_keywords = [
            'bullish', 'surge', 'pump', 'moon', 'breakthrough', 'adoption',
            'partnership', 'upgrade', 'launch', 'success', 'rally', 'gain',
            'rise', 'increase', 'positive', 'optimistic', 'buy', 'invest'
        ]
        
        negative_keywords = [
            'bearish', 'dump', 'crash', 'drop', 'fall', 'decline', 'sell',
            'negative', 'concern', 'risk', 'ban', 'regulation', 'hack',
            'scam', 'fraud', 'loss', 'fear', 'panic', 'bubble', 'correction'
        ]
        
        for article in news:
            text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            
            positive_count = sum(1 for word in positive_keywords if word in text)
            negative_count = sum(1 for word in negative_keywords if word in text)
            
            if positive_count > negative_count:
                sentiment = 'positive'
                score = min(1.0, (positive_count - negative_count) / 10)
            elif negative_count > positive_count:
                sentiment = 'negative'
                score = max(-1.0, (positive_count - negative_count) / 10)
            else:
                sentiment = 'neutral'
                score = 0.0
            
            article['sentiment'] = sentiment
            article['sentiment_score'] = score
        
        return news
    
    def _calculate_overall_sentiment(self, news: List[Dict]) -> Dict[str, Any]:
        """Calcular sentiment general del mercado"""
        if not news:
            return {'overall': 'neutral', 'score': 0, 'distribution': {}}
        
        sentiments = [article.get('sentiment', 'neutral') for article in news]
        scores = [article.get('sentiment_score', 0) for article in news]
        
        sentiment_distribution = Counter(sentiments)
        average_score = sum(scores) / len(scores) if scores else 0
        
        # Determinar sentiment general
        if average_score > 0.2:
            overall = 'positive'
        elif average_score < -0.2:
            overall = 'negative'
        else:
            overall = 'neutral'
        
        return {
            'overall': overall,
            'score': average_score,
            'distribution': dict(sentiment_distribution),
            'total_articles': len(news)
        }
    
    def _generate_news_summary(self, news: List[Dict]) -> Dict[str, Any]:
        """Generar resumen de noticias por crypto"""
        summary = {
            'by_currency': {},
            'trending_topics': [],
            'recent_headlines': []
        }
        
        # Agrupar por moneda
        for article in news:
            currencies = article.get('currencies', [])
            for currency in currencies:
                if currency not in summary['by_currency']:
                    summary['by_currency'][currency] = {
                        'count': 0,
                        'positive': 0,
                        'negative': 0,
                        'recent_headlines': []
                    }
                
                summary['by_currency'][currency]['count'] += 1
                sentiment = article.get('sentiment', 'neutral')
                
                if sentiment == 'positive':
                    summary['by_currency'][currency]['positive'] += 1
                elif sentiment == 'negative':
                    summary['by_currency'][currency]['negative'] += 1
                
                # Agregar headlines recientes
                if len(summary['by_currency'][currency]['recent_headlines']) < 3:
                    summary['by_currency'][currency]['recent_headlines'].append({
                        'title': article.get('title', '')[:100],
                        'sentiment': sentiment,
                        'source': article.get('source', '')
                    })
        
        # Headlines recientes generales
        recent_news = sorted(news, key=lambda x: x.get('published_at', ''), reverse=True)[:5]
        summary['recent_headlines'] = [
            {
                'title': article.get('title', ''),
                'sentiment': article.get('sentiment', 'neutral'),
                'source': article.get('source', ''),
                'currencies': article.get('currencies', [])
            }
            for article in recent_news
        ]
        
        return summary

class SentimentAggregator:
    """Agregador de sentiment desde múltiples fuentes"""
    
    def __init__(self):
        self.fear_greed_api = "https://api.alternative.me/fng/"
    
    def get_market_sentiment(self, news_analyzer: NewsAnalyzer = None) -> Dict[str, Any]:
        """Obtener sentiment general del mercado"""
        sentiment_data = {
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }
        
        # Fear & Greed Index
        try:
            fg_response = requests.get(self.fear_greed_api, timeout=10)
            if fg_response.status_code == 200:
                fg_data = fg_response.json()['data'][0]
                sentiment_data['sources']['fear_greed'] = {
                    'value': int(fg_data['value']),
                    'classification': fg_data['value_classification'],
                    'last_update': fg_data['timestamp']
                }
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed: {e}")
        
        # News sentiment
        if news_analyzer:
            try:
                news_data = news_analyzer.get_crypto_news()
                sentiment_data['sources']['news'] = news_data['sentiment_score']
                sentiment_data['news_summary'] = news_data['summary']
            except Exception as e:
                logger.error(f"Error fetching news sentiment: {e}")
        
        # Calcular sentiment agregado
        sentiment_data['aggregated'] = self._calculate_aggregated_sentiment(sentiment_data['sources'])
        
        return sentiment_data
    
    def _calculate_aggregated_sentiment(self, sources: Dict) -> Dict[str, Any]:
        """Calcular sentiment agregado desde múltiples fuentes"""
        total_weight = 0
        weighted_score = 0
        
        # Fear & Greed Index (peso: 0.4)
        if 'fear_greed' in sources:
            fg_value = sources['fear_greed']['value']
            # Convertir 0-100 a -1 to 1 scale
            normalized_fg = (fg_value - 50) / 50
            weighted_score += normalized_fg * 0.4
            total_weight += 0.4
        
        # News sentiment (peso: 0.6)
        if 'news' in sources and isinstance(sources['news'], dict) and 'score' in sources['news']:
            news_score = sources['news']['score']
            weighted_score += news_score * 0.6
            total_weight += 0.6
        
        if total_weight == 0:
            return {'score': 0, 'classification': 'neutral', 'confidence': 'low'}
        
        final_score = weighted_score / total_weight
        
        # Clasificar
        if final_score > 0.3:
            classification = 'very_positive'
        elif final_score > 0.1:
            classification = 'positive'
        elif final_score < -0.3:
            classification = 'very_negative'
        elif final_score < -0.1:
            classification = 'negative'
        else:
            classification = 'neutral'
        
        # Determinar confianza basada en número de fuentes
        confidence = 'high' if total_weight > 0.7 else 'medium' if total_weight > 0.3 else 'low'
        
        return {
            'score': final_score,
            'classification': classification,
            'confidence': confidence,
            'sources_count': len(sources)
        }

def format_sentiment_for_ai(sentiment_data: Dict, symbol: str = None) -> str:
    """Formatear datos de sentiment para contexto de IA"""
    try:
        output = "ANÁLISIS DE SENTIMENT DEL MERCADO:\n\n"
        
        # Sentiment agregado
        aggregated = sentiment_data.get('aggregated', {})
        if aggregated:
            score = aggregated.get('score', 0)
            classification = aggregated.get('classification', 'neutral').replace('_', ' ').title()
            confidence = aggregated.get('confidence', 'unknown')
            
            output += f"Sentiment general: {classification} (Score: {score:.2f})\n"
            output += f"Confianza: {confidence}\n\n"
        
        # Fear & Greed Index
        sources = sentiment_data.get('sources', {})
        if 'fear_greed' in sources:
            fg = sources['fear_greed']
            output += f"Fear & Greed Index: {fg.get('value', 0)}/100 - {fg.get('classification', 'N/A')}\n"
        
        # News sentiment
        if 'news_summary' in sentiment_data:
            news_summary = sentiment_data['news_summary']
            overall = news_summary.get('overall', 'neutral')
            distribution = news_summary.get('distribution', {})
            total = news_summary.get('total_articles', 0)
            
            if total > 0:
                output += f"\nAnálisis de {total} noticias recientes:\n"
                output += f"- Positivas: {distribution.get('positive', 0)} ({distribution.get('positive', 0)/total*100:.0f}%)\n"
                output += f"- Negativas: {distribution.get('negative', 0)} ({distribution.get('negative', 0)/total*100:.0f}%)\n"
                output += f"- Neutrales: {distribution.get('neutral', 0)} ({distribution.get('neutral', 0)/total*100:.0f}%)\n"
        
        # Headlines recientes específicas del símbolo
        news_summary = sentiment_data.get('news_summary', {})
        if symbol and news_summary.get('by_currency', {}).get(symbol):
            currency_data = news_summary['by_currency'][symbol]
            output += f"\nNoticias específicas de {symbol}:\n"
            for headline in currency_data.get('recent_headlines', [])[:3]:
                sentiment_emoji = "📈" if headline['sentiment'] == 'positive' else "📉" if headline['sentiment'] == 'negative' else "➡️"
                output += f"{sentiment_emoji} {headline['title']}\n"
        
        return output
        
    except Exception as e:
        return f"Error formateando sentiment: {str(e)}"