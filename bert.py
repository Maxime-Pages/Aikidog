from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from bs4 import BeautifulSoup
import requests
import re
import logging
from urllib.parse import quote_plus
import time
from typing import List, Dict, Optional
import threading
from functools import lru_cache
import json
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ImprovedBERTSearchServer:
    def __init__(self):
        """Initialize the enhanced BERT QA pipeline and search components"""
        logger.info("Initializing enhanced BERT model...")
        
        # Initialize BERT QA pipeline with a more capable model
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            tokenizer=AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad"),
        )
        
        # Specialized search endpoints for different query types
        self.search_endpoints = {
            'weather': {
                'api': 'https://api.openweathermap.org/data/2.5/forecast',
                'fallback': 'https://www.weather.com/weather/today/l/'
            },
            'news': [
                'https://www.reddit.com/r/news.json',
                'https://hacker-news.firebaseio.com/v0/topstories.json'
            ],
            'general': [
                'https://searx.be',
                'https://search.sapti.me'
            ],
            'academic': [
                'https://api.crossref.org/works',
                'https://export.arxiv.org/api/query'
            ]
        }
        
        # Request session with optimized settings
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=5)
        
        # Enhanced knowledge base with more comprehensive content
        self.knowledge_base = self._build_knowledge_base()
        
        logger.info("Enhanced BERT Search Server initialized successfully!")

    def _build_knowledge_base(self) -> Dict[str, str]:
        """Build comprehensive knowledge base for fallback answers"""
        return {
            'machine learning': '''Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that build mathematical models based on training data to make predictions or decisions. Key types include supervised learning (with labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through interaction with an environment). Common applications include image recognition, natural language processing, recommendation systems, and predictive analytics.''',
            
            'artificial intelligence': '''Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning (acquiring information and rules), reasoning (using rules to reach conclusions), and self-correction. AI can be categorized as narrow AI (designed for specific tasks) or general AI (matching human cognitive abilities). Current AI applications include virtual assistants, autonomous vehicles, medical diagnosis, financial trading, and content recommendation systems.''',
            
            'deep learning': '''Deep learning is a subset of machine learning based on artificial neural networks with multiple layers (hence "deep"). These neural networks attempt to simulate the behavior of the human brain to learn from large amounts of data. Deep learning has revolutionized fields like computer vision, natural language processing, and speech recognition. Key architectures include convolutional neural networks (CNNs) for image processing, recurrent neural networks (RNNs) for sequential data, and transformers for language understanding.''',
            
            'python programming': '''Python is a high-level, interpreted programming language known for its simple syntax and readability. Created by Guido van Rossum and first released in 1991, Python supports multiple programming paradigms including procedural, object-oriented, and functional programming. It's widely used in web development, data science, artificial intelligence, automation, and scientific computing. Key features include dynamic typing, extensive standard library, and a large ecosystem of third-party packages available through PyPI.''',
            
            'climate change': '''Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, human activities since the Industrial Revolution have been the main driver of climate change, primarily through burning fossil fuels, deforestation, and industrial processes. This increases greenhouse gas concentrations in the atmosphere, leading to global warming. Effects include rising sea levels, more frequent extreme weather events, ecosystem disruption, and threats to food security.''',
            
            'cryptocurrency': '''Cryptocurrency is a digital or virtual currency secured by cryptography, making it nearly impossible to counterfeit. Most cryptocurrencies are decentralized networks based on blockchain technology—a distributed ledger enforced by a network of computers. Bitcoin, created in 2009, was the first cryptocurrency. Others include Ethereum, Litecoin, and thousands of altcoins. Cryptocurrencies can be used for online purchases, investment, and as a store of value, though they remain volatile and face regulatory challenges.'''
        }

    def classify_query(self, question: str) -> str:
        """Classify the type of query to optimize search strategy"""
        question_lower = question.lower()
        
        # Weather queries
        if any(word in question_lower for word in ['weather', 'temperature', 'rain', 'sunny', 'cloudy', 'forecast', 'climate today', 'tomorrow']):
            return 'weather'
        
        # News queries
        if any(word in question_lower for word in ['news', 'current events', 'today', 'latest', 'breaking', 'recent']):
            return 'news'
        
        # Academic/research queries
        if any(word in question_lower for word in ['research', 'study', 'paper', 'journal', 'academic', 'scientific']):
            return 'academic'
        
        # Technical queries
        if any(word in question_lower for word in ['how to', 'tutorial', 'guide', 'programming', 'code', 'technical']):
            return 'technical'
        
        return 'general'

    def search_web_parallel(self, query: str, query_type: str, max_results: int = 5) -> List[Dict]:
        """Enhanced parallel web search with query-type optimization"""
        results = []
        
        # Select search methods based on query type
        if query_type == 'weather':
            search_methods = [
                self._search_weather_specific,
                self._search_searx
            ]
        elif query_type == 'news':
            search_methods = [
                self._search_news_specific,
                self._search_searx
            ]
        elif query_type == 'academic':
            search_methods = [
                self._search_academic,
                self._search_google_scholar,
            ]
        else:
            search_methods = [
                self._search_searx,
                self._search_wikipedia,
                self._search_bing
            ]
        
        # Execute searches in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_method = {
                executor.submit(method, query, max_results): method 
                for method in search_methods[:3]  # Limit to 3 parallel searches
            }
            
            for future in as_completed(future_to_method, timeout=10):
                method = future_to_method[future]
                try:
                    method_results = future.result(timeout=5)
                    if method_results:
                        results.extend(method_results)
                        logger.info(f"Got {len(method_results)} results from {method.__name__}")
                except Exception as e:
                    logger.error(f"Search method {method.__name__} failed: {str(e)}")
        
        # Remove duplicates and return top results
        seen_urls = set()
        unique_results = []
        for result in results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)
        
        return unique_results[:max_results]

    def _search_weather_specific(self, query: str, max_results: int) -> List[Dict]:
        """Specialized weather search"""
        try:
            # Extract location from query
            location_match = re.search(r'weather.*?(?:in|for|at)\s+([a-zA-Z\s,]+)', query.lower())
            location = location_match.group(1).strip() if location_match else 'current location'
            
            # Try multiple weather sources
            weather_sources = [
                f"https://www.weather.com/weather/today/l/{location}",
                f"https://www.accuweather.com/en/search-locations?query={quote_plus(location)}",
                f"https://openweathermap.org/city/{location}"
            ]
            
            results = []
            for source in weather_sources:
                try:
                    response = self.session.get(source, timeout=8)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Extract weather information
                        weather_info = self._extract_weather_info(soup, location)
                        if weather_info:
                            results.append({
                                'title': f"Weather forecast for {location}",
                                'url': source,
                                'snippet': weather_info
                            })
                            break
                except Exception as e:
                    continue
            
            return results
            
        except Exception as e:
            logger.debug(f"Weather search failed: {str(e)}")
            return []

    def _extract_weather_info(self, soup: BeautifulSoup, location: str) -> str:
        """Extract weather information from HTML"""
        try:
            # Common weather data selectors
            weather_selectors = [
                '.current-weather',
                '.today-weather',
                '[data-testid="CurrentConditionsContainer"]',
                '.weather-card',
                '.current-conditions'
            ]
            
            weather_text = []
            
            for selector in weather_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(strip=True)
                    if len(text) > 20 and any(word in text.lower() for word in ['temperature', 'degrees', 'sunny', 'cloudy', 'rain', 'wind']):
                        weather_text.append(text)
            
            if weather_text:
                return f"Current weather conditions for {location}: " + " ".join(weather_text[:3])
            
            # Fallback: extract any weather-related text
            all_text = soup.get_text()
            weather_sentences = []
            for sentence in all_text.split('.'):
                if any(word in sentence.lower() for word in ['temperature', 'degrees', 'weather', 'forecast']):
                    weather_sentences.append(sentence.strip())
            
            return " ".join(weather_sentences[:3]) if weather_sentences else ""
            
        except Exception:
            return ""

    def _search_news_specific(self, query: str, max_results: int) -> List[Dict]:
        """Specialized news search"""
        try:
            # Try multiple news sources
            news_sources = [
                'https://hacker-news.firebaseio.com/v0/topstories.json',
                'https://www.reddit.com/r/news/.json'
            ]
            
            results = []
            
            # Hacker News API
            try:
                hn_response = self.session.get(news_sources[0], timeout=5)
                if hn_response.status_code == 200:
                    story_ids = hn_response.json()[:10]  # Get top 10 stories
                    
                    for story_id in story_ids[:max_results]:
                        story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                        story_response = self.session.get(story_url, timeout=3)
                        if story_response.status_code == 200:
                            story = story_response.json()
                            if story.get('title') and query.lower() in story.get('title', '').lower():
                                results.append({
                                    'title': story.get('title', ''),
                                    'url': story.get('url', f"https://news.ycombinator.com/item?id={story_id}"),
                                    'snippet': story.get('title', '') + " - " + str(story.get('score', 0)) + " points"
                                })
            except Exception:
                pass
            
            return results
            
        except Exception as e:
            logger.debug(f"News search failed: {str(e)}")
            return []

    def _search_academic(self, query: str, max_results: int) -> List[Dict]:
        """Search academic sources"""
        try:
            # arXiv API search
            arxiv_url = "https://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results
            }
            
            response = self.session.get(arxiv_url, params=params, timeout=10)
            if response.status_code == 200:
                # Parse XML response
                from xml.etree import ElementTree as ET
                root = ET.fromstring(response.content)
                
                results = []
                for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                    title = entry.find('.//{http://www.w3.org/2005/Atom}title')
                    summary = entry.find('.//{http://www.w3.org/2005/Atom}summary')
                    link = entry.find('.//{http://www.w3.org/2005/Atom}link[@type="text/html"]')
                    
                    if title is not None:
                        results.append({
                            'title': title.text.strip(),
                            'url': link.get('href') if link is not None else '',
                            'snippet': summary.text.strip()[:300] + '...' if summary is not None else ''
                        })
                
                return results
                
        except Exception as e:
            logger.debug(f"Academic search failed: {str(e)}")
            return []

    def _search_searx(self, query: str, max_results: int) -> List[Dict]:
        """Enhanced SearX search with multiple instances"""
        searx_instances = [
            'https://searx.be',
            'https://search.sapti.me',
            'https://priv.au',
            'https://searx.tiekoetter.com'
        ]
        
        for instance in searx_instances:
            try:
                search_url = f"{instance}/search"
                params = {
                    'q': query,
                    'format': 'json',
                    'categories': 'general',
                    'engines': 'google,bing'
                }
                
                response = self.session.get(search_url, params=params, timeout=8)
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    for item in data.get('results', [])[:max_results]:
                        results.append({
                            'title': item.get('title', ''),
                            'url': item.get('url', ''),
                            'snippet': item.get('content', '')
                        })
                    
                    if results:
                        return results
                        
            except Exception as e:
                logger.debug(f"SearX instance {instance} failed: {str(e)}")
                continue
                
        return []

    def _search_wikipedia(self, query: str, max_results: int) -> List[Dict]:
        """Enhanced Wikipedia search"""
        try:
            # Wikipedia search API
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': max_results,
                'srprop': 'snippet|titlesnippet|size'
            }
            
            response = self.session.get(search_url, params=params, timeout=8)
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('query', {}).get('search', []):
                    # Get page extract for better context
                    page_title = item.get('title', '')
                    extract_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(page_title)}"
                    
                    try:
                        extract_response = self.session.get(extract_url, timeout=3)
                        if extract_response.status_code == 200:
                            extract_data = extract_response.json()
                            snippet = extract_data.get('extract', item.get('snippet', ''))
                        else:
                            snippet = item.get('snippet', '')
                    except:
                        snippet = item.get('snippet', '')
                    
                    results.append({
                        'title': page_title,
                        'url': f"https://en.wikipedia.org/wiki/{quote_plus(page_title)}",
                        'snippet': snippet.replace('<span class="searchmatch">', '').replace('</span>', '')
                    })
                
                return results
                
        except Exception as e:
            logger.debug(f"Wikipedia search failed: {str(e)}")
            return []

    def _search_bing(self, query: str, max_results: int) -> List[Dict]:
        """Enhanced Bing search with better parsing"""
        try:
            search_url = f"https://www.bing.com/search"
            params = {'q': query, 'count': max_results}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            response = self.session.get(search_url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                return []
                
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Multiple selectors for different Bing layouts
            result_selectors = [
                'li.b_algo',
                '.b_algo',
                '[data-react-checksum] li'
            ]
            
            for selector in result_selectors:
                search_results = soup.select(selector)
                if search_results:
                    break
            
            for result in search_results[:max_results]:
                title_elem = result.select_one('h2 a, h3 a, .b_title a')
                snippet_elem = result.select_one('p, .b_caption p, .b_snippet')
                
                if title_elem:
                    results.append({
                        'title': title_elem.get_text(strip=True),
                        'url': title_elem.get('href', ''),
                        'snippet': snippet_elem.get_text(strip=True) if snippet_elem else ''
                    })
            
            return results
            
        except Exception as e:
            logger.debug(f"Bing search failed: {str(e)}")
            return []

    def _search_google_scholar(self, query: str, max_results: int) -> List[Dict]:
        """Enhanced Google Scholar search"""
        try:
            search_url = f"https://scholar.google.com/scholar"
            params = {'q': query, 'hl': 'en'}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = self.session.get(search_url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                return []
                
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            for result in soup.select('.gs_ri')[:max_results]:
                title_elem = result.select_one('.gs_rt a, .gs_rt')
                snippet_elem = result.select_one('.gs_rs')
                link_elem = result.select_one('.gs_rt a')
                
                if title_elem:
                    results.append({
                        'title': title_elem.get_text(strip=True),
                        'url': link_elem.get('href', '') if link_elem else '',
                        'snippet': snippet_elem.get_text(strip=True) if snippet_elem else ''
                    })
            
            return results
            
        except Exception as e:
            logger.debug(f"Google Scholar search failed: {str(e)}")
            return []

    def fetch_content_enhanced(self, url: str) -> str:
        """Enhanced content fetching with better text extraction"""
        try:
            if not url or url.startswith('internal://'):
                return ""
                
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'advertisement']):
                element.decompose()
            
            # Priority content selectors
            content_selectors = [
                'article',
                'main',
                '.content',
                '.post-content',
                '.entry-content',
                '#content',
                '.article-body',
                '.story-body'
            ]
            
            content_text = []
            
            # Try priority selectors first
            for selector in content_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(separator=' ', strip=True)
                    if len(text) > 100:
                        content_text.append(text)
                        break
                if content_text:
                    break
            
            # Fallback to paragraphs
            if not content_text:
                paragraphs = soup.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        content_text.append(text)
            
            # Final fallback
            if not content_text:
                content_text = [soup.get_text(separator=' ', strip=True)]
            
            full_text = ' '.join(content_text)
            
            # Clean up the text
            full_text = re.sub(r'\s+', ' ', full_text)
            full_text = re.sub(r'[^\w\s\.\,\?\!\;\:\-\(\)]', ' ', full_text)
            
            return full_text[:8000]  # Increased limit for better context
            
        except Exception as e:
            logger.error(f"Enhanced content fetch error for {url}: {str(e)}")
            return ""

    def get_fallback_answer(self, question: str) -> str:
        """Generate fallback answer from knowledge base"""
        question_lower = question.lower()
        
        # Find the most relevant knowledge base entry
        best_match = ""
        max_score = 0
        
        for topic, content in self.knowledge_base.items():
            # Calculate relevance score
            topic_words = topic.split()
            score = sum(1 for word in topic_words if word in question_lower)

            # Boost score for exact matches
            if topic in question_lower:
                score += 5
            
            if score > max_score:
                max_score = score
                best_match = content
        
        if best_match:
            return best_match
        
        # Generic fallback
        return "I found some information related to your question, but couldn't provide a specific answer. Please try rephrasing your question or being more specific."

    def answer_question_enhanced(self, question: str, context: str) -> Dict:
        """Enhanced BERT QA with better preprocessing and fallback"""
        try:
            if not context or len(context.strip()) < 10:
                return {
                    'answer': self.get_fallback_answer(question),
                    'confidence': 0.3,
                    'start': 0,
                    'end': 0,
                    'source': 'fallback'
                }
            
            # Clean and prepare inputs
            question = self.clean_text(question)
            context = self.clean_text(context)
            
            # Optimize context length for better performance
            if len(context) > 3000:
                # Try to find the most relevant part of the context
                sentences = context.split('.')
                relevant_sentences = []
                question_words = set(question.lower().split())
                
                for sentence in sentences:
                    sentence_words = set(sentence.lower().split())
                    if question_words.intersection(sentence_words):
                        relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    context = '. '.join(relevant_sentences[:10])  # Top 10 relevant sentences
                else:
                    context = context[:3000]  # Fallback to truncation
            
            # Get answer from BERT
            result = self.qa_pipeline(
                question=question,
                context=context
            )
            
            # Enhance short answers
            answer = result['answer'].strip()
            confidence = result['score']
            
            # If answer is too short and confidence is low, try to expand
            if len(answer) < 10 and confidence < 0.5:
                # Look for sentences containing the answer
                sentences = context.split('.')
                for sentence in sentences:
                    if answer.lower() in sentence.lower():
                        # Use the full sentence as answer
                        expanded_answer = sentence.strip()
                        if len(expanded_answer) > len(answer):
                            answer = expanded_answer
                            confidence = min(confidence + 0.2, 1.0)  # Slight confidence boost
                        break
            
            return {
                'answer': answer,
                'confidence': confidence,
                'start': result.get('start', 0),
                'end': result.get('end', 0),
                'source': 'bert'
            }
            
        except Exception as e:
            logger.error(f"Enhanced QA error: {str(e)}")
            return {
                'answer': self.get_fallback_answer(question),
                'confidence': 0.2,
                'start': 0,
                'end': 0,
                'source': 'fallback'
            }

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\?\!\;\:\-\(\)\[\]\"\'\/]', ' ', text)
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()

    def search_and_answer_enhanced(self, question: str, max_sources: int = 3) -> Dict:
        """Enhanced main method with better performance and accuracy"""
        start_time = time.time()
        
        # Classify query type for optimized search
        query_type = self.classify_query(question)
        logger.info(f"Classified query '{question}' as type: {query_type}")
        
        # Parallel search with query-type optimization
        search_results = self.search_web_parallel(question, query_type, max_sources * 2)
        
        if not search_results:
            fallback_answer = self.get_fallback_answer(question)
            return {
                'question': question,
                'answer': fallback_answer,
                'confidence': 0.3,
                'sources': [],
                'processing_time': time.time() - start_time,
                'query_type': query_type
            }
        
        # Enhanced context collection with parallel content fetching
        contexts = []
        sources_used = []
        
        # Parallel content fetching
        with ThreadPoolExecutor(max_workers=3) as executor:
            fetch_futures = []
            
            for result in search_results[:max_sources]:
                if result['url'] and result['snippet']:
                    future = executor.submit(self.fetch_content_enhanced, result['url'])
                    fetch_futures.append((future, result))
            
            # Collect results from parallel fetching
            for future, result in fetch_futures:
                try:
                    full_content = future.result(timeout=8)
                    
                    if full_content and len(full_content) > 100:
                        contexts.append(full_content)
                        sources_used.append({
                            'title': result['title'],
                            'url': result['url'],
                            'snippet': result['snippet'][:200] + '...' if len(result['snippet']) > 200 else result['snippet']
                        })
                    else:
                        # Use snippet as fallback
                        contexts.append(result['snippet'])
                        sources_used.append({
                            'title': result['title'],
                            'url': result['url'],
                            'snippet': result['snippet']
                        })
                except Exception as e:
                    logger.debug(f"Content fetch timeout for {result['url']}: {str(e)}")
                    # Use snippet as fallback
                    contexts.append(result['snippet'])
                    sources_used.append({
                        'title': result['title'],
                        'url': result['url'],
                        'snippet': result['snippet']
                    })
        
        if not contexts:
            fallback_answer = self.get_fallback_answer(question)
            return {
                'question': question,
                'answer': fallback_answer,
                'confidence': 0.3,
                'sources': sources_used,
                'processing_time': time.time() - start_time,
                'query_type': query_type
            }
        
        # Smart context combination and answer generation
        best_answer = None
        
        # Try different context strategies
        context_strategies = [
            ('combined', ' '.join(contexts)),  # All contexts combined
            ('longest', max(contexts, key=len) if contexts else ''),  # Longest context
            ('most_relevant', self._find_most_relevant_context(question, contexts))  # Most relevant
        ]
        
        for strategy_name, context in context_strategies:
            if not context:
                continue
                
            answer_result = self.answer_question_enhanced(question, context)
            answer_result['strategy'] = strategy_name
            
            # Select best answer based on confidence and length
            if (best_answer is None or 
                answer_result['confidence'] > best_answer['confidence'] or
                (answer_result['confidence'] >= best_answer['confidence'] * 0.9 and 
                 len(answer_result['answer']) > len(best_answer['answer']))):
                best_answer = answer_result
        
        # Final fallback if no good answer found
        if not best_answer or best_answer['confidence'] < 0.2:
            fallback_answer = self.get_fallback_answer(question)
            best_answer = {
                'answer': fallback_answer,
                'confidence': 0.3,
                'source': 'fallback',
                'strategy': 'fallback'
            }
        
        return {
            'question': question,
            'answer': best_answer['answer'],
            'confidence': best_answer['confidence'],
            'sources': sources_used,
            'processing_time': time.time() - start_time,
            'query_type': query_type,
            'strategy_used': best_answer.get('strategy', 'unknown')
        }

    def _find_most_relevant_context(self, question: str, contexts: List[str]) -> str:
        """Find the most relevant context based on keyword overlap"""
        if not contexts:
            return ""
        
        question_words = set(question.lower().split())
        best_context = ""
        max_relevance = 0
        
        for context in contexts:
            context_words = set(context.lower().split())
            relevance = len(question_words.intersection(context_words))
            
            if relevance > max_relevance:
                max_relevance = relevance
                best_context = context
        
        return best_context if best_context else contexts[0]

# Initialize the enhanced server
enhanced_bert_server = ImprovedBERTSearchServer()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Enhanced BERT Search & QA Server',
        'version': '2.0.0',
        'features': [
            'Parallel search processing',
            'Query type classification',
            'Enhanced content extraction',
            'Multiple answer strategies',
            'Comprehensive knowledge base'
        ]
    })

@app.route('/search', methods=['POST'])
def search_endpoint():
    """Enhanced search and answer endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Missing question parameter'
            }), 400
        
        question = data['question'].strip()
        max_sources = data.get('max_sources', 3)
        
        if not question:
            return jsonify({
                'error': 'Question cannot be empty'
            }), 400
        
        # Validate max_sources
        max_sources = max(1, min(max_sources, 10))  # Limit between 1 and 10
        
        # Process the question with enhanced method
        result = enhanced_bert_server.search_and_answer_enhanced(question, max_sources)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Search endpoint error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/qa', methods=['POST'])
def qa_endpoint():
    """Enhanced direct question-answering endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data or 'context' not in data:
            return jsonify({
                'error': 'Missing question or context parameter'
            }), 400
        
        question = data['question'].strip()
        context = data['context'].strip()
        
        if not question or not context:
            return jsonify({
                'error': 'Question and context cannot be empty'
            }), 400
        
        # Get answer using enhanced BERT method
        result = enhanced_bert_server.answer_question_enhanced(question, context)
        
        return jsonify({
            'question': question,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'source': result.get('source', 'bert')
        })
        
    except Exception as e:
        logger.error(f"QA endpoint error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/classify', methods=['POST'])
def classify_endpoint():
    """Query classification endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Missing question parameter'
            }), 400
        
        question = data['question'].strip()
        
        if not question:
            return jsonify({
                'error': 'Question cannot be empty'
            }), 400
        
        query_type = enhanced_bert_server.classify_query(question)
        
        return jsonify({
            'question': question,
            'query_type': query_type,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Classify endpoint error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/knowledge', methods=['GET'])
def knowledge_endpoint():
    """Knowledge base topics endpoint"""
    try:
        topics = list(enhanced_bert_server.knowledge_base.keys())
        return jsonify({
            'available_topics': topics,
            'total_topics': len(topics)
        })
        
    except Exception as e:
        logger.error(f"Knowledge endpoint error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(429)
def rate_limit_error(error):
    return jsonify({'error': 'Rate limit exceeded, please try again later'}), 429

if __name__ == '__main__':
    print(" Starting Enhanced BERT Search & QA Server...")
    print(" Available Endpoints:")
    print("   GET  / - Health check and features")
    print("   POST /search - Enhanced search web and answer question")
    print("   POST /qa - Answer question with provided context")
    print("   POST /classify - Classify query type")
    print("   GET  /knowledge - List available knowledge base topics")
    print("\n New Features:")
    print("    Parallel search processing for faster results")
    print("    Query type classification (weather, news, academic, etc.)")
    print("    Enhanced content extraction and text cleaning")
    print("    Multiple answer generation strategies")
    print("    Comprehensive knowledge base fallback")
    print("    Better weather and news search capabilities")
    print("    Academic paper search via arXiv")
    print("\n Example usage:")
    print('   curl -X POST http://localhost:5000/search \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"question": "What is the weather in Paris tomorrow?", "max_sources": 5}\'')
    print('\n   curl -X POST http://localhost:5000/classify \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"question": "What is machine learning?"}\'')
    
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)