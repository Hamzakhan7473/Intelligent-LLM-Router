# backend/app/services/llm_service.py

import asyncio
import httpx
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from app.ml.routing.gpt5_router import GPT5VariantRouter, GPTVariant, RoutingResult
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    GOOGLE = "google"
    LOCAL = "local"

@dataclass
class LLMResponse:
    content: str
    model: str
    provider: LLMProvider
    tokens_used: int
    cost: float
    latency_ms: int
    quality_estimate: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class APICallMetrics:
    request_start: float
    request_end: float
    tokens_input: int
    tokens_output: int
    cost_calculated: float
    success: bool
    error_message: Optional[str] = None

class LLMAPIService:
    """
    Service for making API calls to various LLM providers with intelligent routing,
    fallback handling, and performance tracking.
    """
    
    def __init__(self):
        self.router = GPT5VariantRouter()
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # API configurations
        self.api_configs = {
            LLMProvider.OPENAI: {
                "base_url": "https://api.openai.com/v1",
                "headers": {
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
            },
            LLMProvider.ANTHROPIC: {
                "base_url": "https://api.anthropic.com/v1", 
                "headers": {
                    "x-api-key": settings.ANTHROPIC_API_KEY,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
            },
            LLMProvider.GOOGLE: {
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "headers": {
                    "Content-Type": "application/json"
                },
                "params": {
                    "key": settings.GOOGLE_API_KEY
                }
            }
        }
        
        # Model mappings to actual API model names
        self.model_mappings = {
            GPTVariant.GPT5_NANO: {
                LLMProvider.OPENAI: "gpt-5-nano",
                LLMProvider.ANTHROPIC: "claude-3-haiku-20240307",  # Fallback
                LLMProvider.GOOGLE: "gemini-1.5-flash"  # Fallback
            },
            GPTVariant.GPT5_MINI: {
                LLMProvider.OPENAI: "gpt-5-mini", 
                LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
                LLMProvider.GOOGLE: "gemini-1.5-pro"
            },
            GPTVariant.GPT5_CHAT: {
                LLMProvider.OPENAI: "gpt-5-chat",
                LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022", 
                LLMProvider.GOOGLE: "gemini-1.5-pro"
            },
            GPTVariant.GPT5_FULL: {
                LLMProvider.OPENAI: "gpt-5",
                LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
                LLMProvider.GOOGLE: "gemini-1.5-pro"
            }
        }
        
        # Fallback chains for when models are unavailable
        self.fallback_chains = {
            GPTVariant.GPT5_FULL: [
                (LLMProvider.OPENAI, "gpt-5"),
                (LLMProvider.OPENAI, "gpt-5-chat"),
                (LLMProvider.ANTHROPIC, "claude-3-5-sonnet-20241022"),
                (LLMProvider.OPENAI, "gpt-4o")
            ],
            GPTVariant.GPT5_CHAT: [
                (LLMProvider.OPENAI, "gpt-5-chat"),
                (LLMProvider.OPENAI, "gpt-5"),
                (LLMProvider.ANTHROPIC, "claude-3-5-sonnet-20241022"),
                (LLMProvider.OPENAI, "gpt-4o")
            ],
            GPTVariant.GPT5_MINI: [
                (LLMProvider.OPENAI, "gpt-5-mini"),
                (LLMProvider.OPENAI, "gpt-5-nano"),
                (LLMProvider.OPENAI, "gpt-4o-mini"),
                (LLMProvider.ANTHROPIC, "claude-3-haiku-20240307")
            ],
            GPTVariant.GPT5_NANO: [
                (LLMProvider.OPENAI, "gpt-5-nano"),
                (LLMProvider.OPENAI, "gpt-5-mini"),
                (LLMProvider.OPENAI, "gpt-3.5-turbo"),
                (LLMProvider.ANTHROPIC, "claude-3-haiku-20240307")
            ]
        }
    
    async def route_and_call(self, prompt: str, user_preferences: Dict[str, Any],
                           context: Optional[Dict[str, Any]] = None) -> Tuple[LLMResponse, RoutingResult]:
        """
        Main method: Routes prompt to optimal model and makes API call
        """
        # Step 1: Get routing decision
        routing_result = await self.router.route_with_analysis(prompt, user_preferences)
        
        logger.info(f"Routing decision: {routing_result.selected_variant.value} "
                   f"(complexity: {routing_result.complexity_score:.2f}, "
                   f"confidence: {routing_result.confidence:.2f})")
        
        # Step 2: Make API call with fallback handling
        llm_response = await self.call_llm_with_fallback(
            variant=routing_result.selected_variant,
            prompt=prompt,
            context=context
        )
        
        # Step 3: Log performance metrics
        await self.log_performance_metrics(routing_result, llm_response, prompt)
        
        return llm_response, routing_result
    
    async def call_llm_with_fallback(self, variant: GPTVariant, prompt: str,
                                   context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Call LLM API with intelligent fallback handling
        """
        fallback_chain = self.fallback_chains.get(variant, [])
        
        for provider, model_name in fallback_chain:
            try:
                logger.info(f"Attempting API call to {provider.value}:{model_name}")
                
                response = await self.make_api_call(
                    provider=provider,
                    model=model_name,
                    prompt=prompt,
                    context=context
                )
                
                if response:
                    logger.info(f"Successful API call to {provider.value}:{model_name}")
                    return response
                    
            except Exception as e:
                logger.warning(f"API call failed for {provider.value}:{model_name} - {str(e)}")
                continue
        
        # All fallbacks failed
        raise Exception(f"All API calls failed for variant {variant.value}")
    
    async def make_api_call(self, provider: LLMProvider, model: str, 
                           prompt: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Make actual API call to specific provider
        """
        start_time = time.time()
        
        try:
            if provider == LLMProvider.OPENAI:
                response = await self.call_openai_api(model, prompt, context)
            elif provider == LLMProvider.ANTHROPIC:
                response = await self.call_anthropic_api(model, prompt, context)
            elif provider == LLMProvider.GOOGLE:
                response = await self.call_google_api(model, prompt, context)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            # Add timing information
            response.latency_ms = latency_ms
            response.timestamp = datetime.now()
            
            return response
            
        except Exception as e:
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            logger.error(f"API call failed after {latency_ms}ms: {str(e)}")
            raise
    
    async def call_openai_api(self, model: str, prompt: str, 
                            context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Call OpenAI API (GPT models)
        """
        config = self.api_configs[LLMProvider.OPENAI]
        
        # Prepare messages
        messages = []
        if context and context.get('system_message'):
            messages.append({
                "role": "system",
                "content": context['system_message']
            })
        
        messages.append({
            "role": "user", 
            "content": prompt
        })
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": context.get('max_tokens', 4000) if context else 4000,
            "temperature": context.get('temperature', 0.7) if context else 0.7,
            "stream": False
        }
        
        response = await self.client.post(
            f"{config['base_url']}/chat/completions",
            headers=config['headers'],
            json=payload
        )
        
        response.raise_for_status()
        data = response.json()
        
        # Extract response data
        content = data['choices'][0]['message']['content']
        tokens_used = data['usage']['total_tokens']
        input_tokens = data['usage']['prompt_tokens'] 
        output_tokens = data['usage']['completion_tokens']
        
        # Calculate cost based on model
        cost = self.calculate_openai_cost(model, input_tokens, output_tokens)
        
        return LLMResponse(
            content=content,
            model=model,
            provider=LLMProvider.OPENAI,
            tokens_used=tokens_used,
            cost=cost,
            latency_ms=0,  # Will be set by caller
            quality_estimate=self.estimate_quality(content, prompt),
            timestamp=datetime.now(),
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "finish_reason": data['choices'][0].get('finish_reason'),
                "model_version": data.get('model')
            }
        )
    
    async def call_anthropic_api(self, model: str, prompt: str,
                               context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Call Anthropic API (Claude models) - Fallback option
        """
        config = self.api_configs[LLMProvider.ANTHROPIC]
        
        payload = {
            "model": model,
            "max_tokens": context.get('max_tokens', 4000) if context else 4000,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": context.get('temperature', 0.7) if context else 0.7
        }
        
        if context and context.get('system_message'):
            payload['system'] = context['system_message']
        
        response = await self.client.post(
            f"{config['base_url']}/messages",
            headers=config['headers'],
            json=payload
        )
        
        response.raise_for_status()
        data = response.json()
        
        content = data['content'][0]['text']
        input_tokens = data['usage']['input_tokens']
        output_tokens = data['usage']['output_tokens']
        tokens_used = input_tokens + output_tokens
        
        cost = self.calculate_anthropic_cost(model, input_tokens, output_tokens)
        
        return LLMResponse(
            content=content,
            model=model,
            provider=LLMProvider.ANTHROPIC,
            tokens_used=tokens_used,
            cost=cost,
            latency_ms=0,
            quality_estimate=self.estimate_quality(content, prompt),
            timestamp=datetime.now(),
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "stop_reason": data.get('stop_reason'),
                "model_version": data.get('model')
            }
        )
    
    async def call_google_api(self, model: str, prompt: str,
                            context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Call Google Gemini API - Fallback option
        """
        config = self.api_configs[LLMProvider.GOOGLE]
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": context.get('max_tokens', 4000) if context else 4000,
                "temperature": context.get('temperature', 0.7) if context else 0.7
            }
        }
        
        url = f"{config['base_url']}/models/{model}:generateContent"
        
        response = await self.client.post(
            url,
            headers=config['headers'],
            json=payload,
            params=config.get('params', {})
        )
        
        response.raise_for_status()
        data = response.json()
        
        content = data['candidates'][0]['content']['parts'][0]['text']
        
        # Google doesn't provide token counts in the same way, estimate
        estimated_tokens = len(prompt.split()) + len(content.split())
        cost = estimated_tokens * 0.001  # Rough estimate
        
        return LLMResponse(
            content=content,
            model=model,
            provider=LLMProvider.GOOGLE,
            tokens_used=estimated_tokens,
            cost=cost,
            latency_ms=0,
            quality_estimate=self.estimate_quality(content, prompt),
            timestamp=datetime.now(),
            metadata={
                "finish_reason": data['candidates'][0].get('finishReason'),
                "safety_ratings": data['candidates'][0].get('safetyRatings', [])
            }
        )
    
    def calculate_openai_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for OpenAI models"""
        # Updated pricing for GPT-5 variants (as of Aug 2025)
        pricing = {
            "gpt-5": {"input": 0.015, "output": 0.020},
            "gpt-5-chat": {"input": 0.012, "output": 0.016},
            "gpt-5-mini": {"input": 0.008, "output": 0.010}, 
            "gpt-5-nano": {"input": 0.003, "output": 0.004},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
        
        model_pricing = pricing.get(model, pricing["gpt-4o"])  # Default fallback
        
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"] 
        
        return input_cost + output_cost
    
    def calculate_anthropic_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Anthropic models"""
        pricing = {
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125}
        }
        
        model_pricing = pricing.get(model, pricing["claude-3-5-sonnet-20241022"])
        
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    def estimate_quality(self, response: str, prompt: str) -> float:
        """
        Estimate response quality based on heuristics
        This is a simple implementation - could be enhanced with ML models
        """
        quality_score = 5.0  # Base score
        
        # Length appropriateness
        response_words = len(response.split())
        prompt_words = len(prompt.split())
        
        if response_words < prompt_words * 0.5:
            quality_score -= 1.0  # Too short
        elif response_words > prompt_words * 10:
            quality_score -= 0.5  # Possibly too verbose
        
        # Structure indicators
        if any(indicator in response.lower() for indicator in 
               ['step 1', 'first', 'second', 'finally', '1.', '2.']):
            quality_score += 0.5  # Well-structured
            
        # Code quality (if applicable)
        if '```' in response:
            quality_score += 0.5  # Contains code blocks
            
        # Completeness indicators
        if any(indicator in response.lower() for indicator in
               ['in conclusion', 'to summarize', 'overall']):
            quality_score += 0.3  # Has conclusion
            
        return min(max(quality_score, 1.0), 10.0)  # Clamp between 1-10
    
    async def log_performance_metrics(self, routing_result: RoutingResult, 
                                    llm_response: LLMResponse, prompt: str):
        """
        Log performance metrics for analysis and optimization
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "prompt_hash": hash(prompt) % 100000,  # Anonymized
            "prompt_length": len(prompt),
            "selected_variant": routing_result.selected_variant.value,
            "complexity_score": routing_result.complexity_score,
            "routing_confidence": routing_result.confidence,
            "actual_model": llm_response.model,
            "actual_provider": llm_response.provider.value,
            "actual_cost": llm_response.cost,
            "actual_latency_ms": llm_response.latency_ms,
            "tokens_used": llm_response.tokens_used,
            "quality_estimate": llm_response.quality_estimate,
            "estimated_cost": routing_result.estimated_cost,
            "estimated_latency": routing_result.estimated_latency,
            "cost_accuracy": abs(llm_response.cost - routing_result.estimated_cost),
            "latency_accuracy": abs(llm_response.latency_ms - routing_result.estimated_latency)
        }
        
        # In production, this would go to a database or monitoring system
        logger.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")
    
    async def get_model_availability(self) -> Dict[str, bool]:
        """
        Check which models are currently available
        """
        availability = {}
        
        # Test each provider with a simple prompt
        test_prompt = "Hello"
        
        for provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GOOGLE]:
            try:
                # Try a simple call
                if provider == LLMProvider.OPENAI:
                    await self.call_openai_api("gpt-5-nano", test_prompt)
                    availability[provider.value] = True
                elif provider == LLMProvider.ANTHROPIC:
                    await self.call_anthropic_api("claude-3-haiku-20240307", test_prompt)
                    availability[provider.value] = True
                elif provider == LLMProvider.GOOGLE:
                    await self.call_google_api("gemini-1.5-flash", test_prompt)
                    availability[provider.value] = True
            except:
                availability[provider.value] = False
        
        return availability
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()


# Example usage and testing
async def test_llm_service():
    """
    Test the LLM service with various prompts and priorities
    """
    service = LLMAPIService()
    
    test_cases = [
        {
            "prompt": "What's 2+2?",
            "preferences": {"priority": "speed", "max_cost": 0.01}
        },
        {
            "prompt": "Write a Python function to implement bubble sort with detailed comments",
            "preferences": {"priority": "cost", "max_cost": 0.02}
        },
        {
            "prompt": "Analyze the philosophical implications of artificial intelligence on human consciousness, considering multiple perspectives and providing a nuanced conclusion",
            "preferences": {"priority": "quality"}
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"\n=== Test Case {i} ===")
            print(f"Prompt: {test_case['prompt']}")
            print(f"Preferences: {test_case['preferences']}")
            
            response, routing = await service.route_and_call(
                test_case['prompt'], 
                test_case['preferences']
            )
            
            print(f"\nRouting Decision:")
            print(f"  Model: {routing.selected_variant.value}")
            print(f"  Complexity: {routing.complexity_score:.2f}")
            print(f"  Confidence: {routing.confidence:.2f}")
            print(f"  Reasoning: {routing.reasoning}")
            
            print(f"\nAPI Response:")
            print(f"  Provider: {response.provider.value}")
            print(f"  Actual Model: {response.model}")
            print(f"  Cost: ${response.cost:.4f}")
            print(f"  Latency: {response.latency_ms}ms")
            print(f"  Tokens: {response.tokens_used}")
            print(f"  Quality: {response.quality_estimate:.1f}/10")
            print(f"  Response: {response.content[:100]}...")
            
        except Exception as e:
            print(f"Error in test case {i}: {str(e)}")
    
    await service.client.aclose()

if __name__ == "__main__":
    asyncio.run(test_llm_service())