import openai
import anthropic
from typing import List, Dict, Any
from app.models.chat import ChatMessage, ChatUsage
from app.core.config import settings
from datetime import datetime
import asyncio
import httpx

class ChatService:
    def __init__(self):
        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None
        
        if settings.OPENAI_API_KEY:
            self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        if settings.ANTHROPIC_API_KEY:
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    
    async def generate_response(
        self, 
        messages: List[ChatMessage], 
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate response using specified model and provider
        """
        try:
            # Determine provider based on model
            provider = self._get_provider_for_model(model)
            
            if provider == "openai":
                return await self._generate_openai_response(messages, model, temperature, max_tokens)
            elif provider == "anthropic":
                return await self._generate_anthropic_response(messages, model, temperature, max_tokens)
            else:
                raise ValueError(f"Unsupported model: {model}")
                
        except Exception as e:
            raise Exception(f"Response generation failed: {str(e)}")
    
    async def _generate_openai_response(
        self, 
        messages: List[ChatMessage], 
        model: str, 
        temperature: float, 
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        if not self.openai_client:
            raise Exception("OpenAI API key not configured")
        
        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        
        response = await self.openai_client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        assistant_message = ChatMessage(
            role="assistant",
            content=response.choices[0].message.content,
            timestamp=datetime.now()
        )
        
        usage = ChatUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
        
        return {
            "message": assistant_message,
            "usage": usage,
            "provider": "openai"
        }
    
    async def _generate_anthropic_response(
        self, 
        messages: List[ChatMessage], 
        model: str, 
        temperature: float, 
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate response using Anthropic API"""
        if not self.anthropic_client:
            raise Exception("Anthropic API key not configured")
        
        # Convert messages to Anthropic format
        system_message = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        response = await self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message if system_message else "You are a helpful AI assistant.",
            messages=anthropic_messages
        )
        
        assistant_message = ChatMessage(
            role="assistant",
            content=response.content[0].text,
            timestamp=datetime.now()
        )
        
        usage = ChatUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens
        )
        
        return {
            "message": assistant_message,
            "usage": usage,
            "provider": "anthropic"
        }
    
    def _get_provider_for_model(self, model: str) -> str:
        """Determine provider based on model name"""
        if model.startswith("gpt-") or model.startswith("text-"):
            return "openai"
        elif model.startswith("claude-"):
            return "anthropic"
        else:
            # Default to OpenAI for unknown models
            return "openai"
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from all providers"""
        models = []
        
        # OpenAI models
        if self.openai_client:
            openai_models = [
                {"name": "gpt-4", "provider": "openai", "type": "chat", "description": "Most capable GPT-4 model"},
                {"name": "gpt-4-turbo", "provider": "openai", "type": "chat", "description": "Faster GPT-4 variant"},
                {"name": "gpt-3.5-turbo", "provider": "openai", "type": "chat", "description": "Fast and efficient model"},
            ]
            models.extend(openai_models)
        
        # Anthropic models
        if self.anthropic_client:
            anthropic_models = [
                {"name": "claude-3-opus-20240229", "provider": "anthropic", "type": "chat", "description": "Most capable Claude model"},
                {"name": "claude-3-sonnet-20240229", "provider": "anthropic", "type": "chat", "description": "Balanced performance and speed"},
                {"name": "claude-3-haiku-20240307", "provider": "anthropic", "type": "chat", "description": "Fastest Claude model"},
            ]
            models.extend(anthropic_models)
        
        return models