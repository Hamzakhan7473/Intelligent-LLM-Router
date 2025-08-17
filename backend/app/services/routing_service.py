from typing import List, Dict, Any, Optional
from app.models.chat import ChatMessage
from app.core.config import settings
import re
import asyncio

class RoutingService:
    def __init__(self):
        # Model capabilities matrix - updated with correct OpenAI model names
        self.model_capabilities = {
            # OpenAI Models (using correct names)
            "gpt-4": {
                "provider": "openai",
                "speed": 6,  # 1-10 scale
                "quality": 10,
                "cost": 3,  # 1-10 scale (lower is cheaper)
                "max_tokens": 8192,
                "strengths": ["reasoning", "analysis", "complex_tasks", "coding"],
                "ideal_for": ["analytical", "coding", "creative"]
            },
            "gpt-3.5-turbo": {
                "provider": "openai",
                "speed": 9,
                "quality": 7,
                "cost": 8,
                "max_tokens": 4096,
                "strengths": ["speed", "cost_effective", "general_chat"],
                "ideal_for": ["chat", "simple_tasks", "coding", "analytical", "creative"]
            },
            
            # Anthropic Models (optional - only if you have Anthropic API key)
            "claude-3-sonnet-20240229": {
                "provider": "anthropic",
                "speed": 7,
                "quality": 8,
                "cost": 6,
                "max_tokens": 4096,
                "strengths": ["balanced", "reliable", "good_reasoning"],
                "ideal_for": ["chat", "analytical", "coding"]
            }
        }
    
    async def select_optimal_model(
        self,
        messages: List[ChatMessage],
        task_type: str = "chat",
        prefer_speed: bool = False,
        prefer_quality: bool = True
    ) -> Dict[str, Any]:
        """
        Intelligently select the optimal model based on conversation context and preferences
        """
        try:
            # For now, let's default to gpt-3.5-turbo which everyone has access to
            # You can make this more sophisticated later
            selected_model = "gpt-3.5-turbo"
            
            # Simple routing logic
            context_analysis = self._analyze_conversation_context(messages)
            
            if context_analysis["has_code"] and prefer_quality:
                # Use GPT-4 for complex coding if available, otherwise fallback
                selected_model = "gpt-4" if self._model_available("gpt-4") else "gpt-3.5-turbo"
            elif context_analysis["is_creative"] and prefer_quality:
                selected_model = "gpt-4" if self._model_available("gpt-4") else "gpt-3.5-turbo"
            else:
                selected_model = "gpt-3.5-turbo"  # Safe default
            
            reasoning = f"Selected {selected_model} for {task_type} task"
            if context_analysis["has_code"]:
                reasoning += " (coding detected)"
            if context_analysis["is_creative"]:
                reasoning += " (creative task detected)"
                
            return {
                "model": selected_model,
                "provider": self.model_capabilities[selected_model]["provider"],
                "reasoning": reasoning,
                "confidence": 0.8,
                "alternatives": [{"model": "gpt-3.5-turbo", "score": 0.7}]
            }
            
        except Exception as e:
            # Fallback to safe default
            return {
                "model": "gpt-3.5-turbo",
                "provider": "openai", 
                "reasoning": f"Fallback to default model due to routing error: {str(e)}",
                "confidence": 0.5,
                "alternatives": []
            }
    
    def _model_available(self, model: str) -> bool:
        """Check if a model is available (simplified check)"""
        # For now, assume gpt-3.5-turbo is always available
        # You can add API calls to check model availability later
        return model == "gpt-3.5-turbo"
    
    def _analyze_conversation_context(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Analyze conversation to understand context and requirements"""
        analysis = {
            "message_count": len(messages),
            "total_length": sum(len(msg.content) for msg in messages),
            "has_code": False,
            "is_creative": False,
            "is_analytical": False,
            "complexity_score": 0,
        }
        
        latest_message = messages[-1].content.lower() if messages else ""
        all_content = " ".join([msg.content.lower() for msg in messages])
        
        # Detect code patterns
        code_patterns = [r'```', r'def ', r'function', r'class ', r'import ', r'<.*>', r'\{.*\}']
        analysis["has_code"] = any(re.search(pattern, all_content) for pattern in code_patterns)
        
        # Detect creative requests
        creative_keywords = ['write', 'story', 'poem', 'creative', 'imagine', 'fiction', 'character']
        analysis["is_creative"] = any(keyword in all_content for keyword in creative_keywords)
        
        # Detect analytical requests  
        analytical_keywords = ['analyze', 'compare', 'evaluate', 'explain', 'research', 'study']
        analysis["is_analytical"] = any(keyword in all_content for keyword in analytical_keywords)
        
        return analysis
