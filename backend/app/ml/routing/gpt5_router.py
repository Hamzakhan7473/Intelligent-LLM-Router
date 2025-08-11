# backend/app/ml/routing/gpt5_router.py

import re
import math
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class Priority(Enum):
    SPEED = "speed"
    COST = "cost"
    QUALITY = "quality"

class GPTVariant(Enum):
    GPT5_NANO = "gpt-5-nano"
    GPT5_MINI = "gpt-5-mini" 
    GPT5_CHAT = "gpt-5-chat"
    GPT5_FULL = "gpt-5"

@dataclass
class ModelSpecs:
    name: str
    cost_per_1k_tokens: float
    avg_latency_ms: int
    quality_score: float
    max_tokens: int
    context_window: int
    strengths: List[str]

@dataclass
class PromptFeatures:
    has_code_request: bool = False
    has_reasoning_request: bool = False
    requires_long_context: bool = False
    is_conversational: bool = False
    is_creative: bool = False
    is_technical: bool = False
    quality_sensitive: bool = False
    multi_part: bool = False
    complexity_indicators: int = 0
    word_count: int = 0
    question_count: int = 0

@dataclass
class UserPreferences:
    priority: Priority = Priority.QUALITY
    max_cost: float = 0.02
    max_latency: int = 5000
    speed_weight: float = 0.33
    cost_weight: float = 0.33  
    quality_weight: float = 0.34

@dataclass
class RoutingResult:
    selected_variant: GPTVariant
    complexity_score: float
    confidence: float
    reasoning: str
    estimated_cost: float
    estimated_latency: int
    alternatives: List[Dict[str, Any]]
    performance_prediction: Dict[str, float]

class GPT5VariantRouter:
    """
    Intelligent router for selecting optimal GPT-5 variant based on
    prompt complexity, user preferences, and performance requirements.
    """
    
    def __init__(self):
        self.model_specs = {
            GPTVariant.GPT5_NANO: ModelSpecs(
                name="GPT-5 Nano",
                cost_per_1k_tokens=0.003,
                avg_latency_ms=400,
                quality_score=7.9,
                max_tokens=16000,
                context_window=16000,
                strengths=["simple_qa", "general_chat", "basic_writing", "quick_responses"]
            ),
            GPTVariant.GPT5_MINI: ModelSpecs(
                name="GPT-5 Mini", 
                cost_per_1k_tokens=0.008,
                avg_latency_ms=800,
                quality_score=8.8,
                max_tokens=64000,
                context_window=64000,
                strengths=["summarization", "coding", "analysis", "translation", "documentation"]
            ),
            GPTVariant.GPT5_CHAT: ModelSpecs(
                name="GPT-5 Chat",
                cost_per_1k_tokens=0.012,
                avg_latency_ms=1000, 
                quality_score=9.2,
                max_tokens=128000,
                context_window=128000,
                strengths=["reasoning", "explanations", "step_by_step", "conversational_analysis"]
            ),
            GPTVariant.GPT5_FULL: ModelSpecs(
                name="GPT-5",
                cost_per_1k_tokens=0.015,
                avg_latency_ms=1200,
                quality_score=9.5,
                max_tokens=128000,
                context_window=128000,
                strengths=["complex_reasoning", "app_generation", "research", "creative_writing"]
            )
        }
        
        self.complexity_thresholds = {
            'nano_max': 0.3,      # Simple tasks only
            'mini_max': 0.7,      # Medium complexity
            'chat_max': 0.85,     # Reasoning tasks
            'full_required': 0.9   # Complex tasks requiring full model
        }
        
    def extract_prompt_features(self, prompt: str) -> PromptFeatures:
        """Extract features from prompt text for routing decisions"""
        prompt_lower = prompt.lower()
        
        # Code-related detection
        code_keywords = [
            'function', 'code', 'script', 'program', 'algorithm', 'debug',
            'create app', 'build application', 'api', 'database', 'framework',
            'python', 'javascript', 'react', 'node', 'sql', 'html', 'css'
        ]
        has_code = any(keyword in prompt_lower for keyword in code_keywords)
        
        # Reasoning detection
        reasoning_keywords = [
            'step by step', 'think through', 'analyze', 'compare', 'evaluate',
            'pros and cons', 'because', 'therefore', 'however', 'explain why',
            'reasoning', 'logic', 'conclude', 'infer', 'deduce'
        ]
        has_reasoning = any(keyword in prompt_lower for keyword in reasoning_keywords)
        
        # Technical domain detection
        technical_domains = [
            'machine learning', 'artificial intelligence', 'data science', 'quantum',
            'blockchain', 'cybersecurity', 'distributed systems', 'microservices',
            'devops', 'cloud', 'kubernetes', 'docker', 'tensorflow', 'pytorch'
        ]
        is_technical = any(domain in prompt_lower for domain in technical_domains)
        
        # Creative writing detection
        creative_keywords = [
            'write', 'story', 'poem', 'creative', 'narrative', 'character',
            'plot', 'dialogue', 'fiction', 'novel', 'screenplay'
        ]
        is_creative = any(keyword in prompt_lower for keyword in creative_keywords)
        
        # Conversational detection
        conversational_keywords = [
            'chat', 'discuss', 'talk about', 'conversation', 'tell me',
            'what do you think', 'opinion', 'advice', 'recommend'
        ]
        is_conversational = any(keyword in prompt_lower for keyword in conversational_keywords)
        
        # Quality sensitivity detection
        quality_keywords = [
            'important', 'critical', 'careful', 'precise', 'accurate',
            'detailed', 'thorough', 'comprehensive', 'professional'
        ]
        quality_sensitive = any(keyword in prompt_lower for keyword in quality_keywords)
        
        # Complexity indicators
        complexity_indicators = 0
        complexity_phrases = [
            'complex', 'advanced', 'sophisticated', 'comprehensive', 'detailed',
            'in-depth', 'thorough', 'complete analysis', 'full implementation',
            'enterprise-level', 'production-ready', 'scalable'
        ]
        complexity_indicators = sum(1 for phrase in complexity_phrases if phrase in prompt_lower)
        
        return PromptFeatures(
            has_code_request=has_code,
            has_reasoning_request=has_reasoning,
            requires_long_context=len(prompt) > 8000,
            is_conversational=is_conversational,
            is_creative=is_creative,
            is_technical=is_technical,
            quality_sensitive=quality_sensitive,
            multi_part=prompt.count('?') > 1 or len(prompt.split('.')) > 8,
            complexity_indicators=complexity_indicators,
            word_count=len(prompt.split()),
            question_count=prompt.count('?')
        )
    
    def calculate_complexity_score(self, prompt: str, features: PromptFeatures) -> float:
        """
        Calculate complexity score from 0.0 to 1.0
        Higher score = more complex = needs stronger model
        """
        complexity = 0.0
        
        # Base complexity from length
        length_factor = min(len(prompt) / 3000, 0.25)  # Cap at 0.25
        complexity += length_factor
        
        # Code complexity scoring
        if features.has_code_request:
            app_generation_keywords = [
                'create app', 'build application', 'full stack', 'complete system',
                'end-to-end', 'entire application', 'production app'
            ]
            if any(keyword in prompt.lower() for keyword in app_generation_keywords):
                complexity += 0.4  # High complexity for app generation
            elif any(keyword in prompt.lower() for keyword in ['algorithm', 'optimize', 'performance', 'scalable']):
                complexity += 0.3  # Advanced coding
            else:
                complexity += 0.2  # Basic coding
        
        # Reasoning complexity
        if features.has_reasoning_request:
            multi_step_keywords = ['first', 'then', 'next', 'finally', 'step 1', 'step 2']
            if any(keyword in prompt.lower() for keyword in multi_step_keywords):
                complexity += 0.35  # Multi-step reasoning
            else:
                complexity += 0.25  # Basic reasoning
        
        # Technical domain bonus
        if features.is_technical:
            complexity += 0.2
            
        # Multi-part questions
        if features.multi_part:
            part_bonus = min(features.question_count * 0.1, 0.3)
            complexity += part_bonus
        
        # Complexity indicators from language
        indicator_bonus = min(features.complexity_indicators * 0.08, 0.25)
        complexity += indicator_bonus
        
        # Context requirements
        if features.requires_long_context:
            complexity += 0.15
            
        # Quality sensitivity (needs more capable model)
        if features.quality_sensitive:
            complexity += 0.1
            
        # Creative writing (often complex)
        if features.is_creative and features.word_count > 50:
            complexity += 0.15
            
        return min(complexity, 1.0)  # Cap at 1.0
    
    def route_by_priority(self, complexity: float, features: PromptFeatures, 
                         priority: Priority) -> GPTVariant:
        """Route based on user's priority preference"""
        
        if priority == Priority.SPEED:
            return self._route_for_speed(complexity, features)
        elif priority == Priority.COST:
            return self._route_for_cost(complexity, features)
        else:  # QUALITY
            return self._route_for_quality(complexity, features)
    
    def _route_for_speed(self, complexity: float, features: PromptFeatures) -> GPTVariant:
        """Optimize for fastest response time"""
        # Force full model for very complex tasks
        if complexity > 0.9:
            return GPTVariant.GPT5_FULL
        
        # Use nano for simple tasks
        if complexity < self.complexity_thresholds['nano_max']:
            return GPTVariant.GPT5_NANO
            
        # Use mini for medium complexity
        if complexity < self.complexity_thresholds['mini_max']:
            return GPTVariant.GPT5_MINI
            
        # Use chat for reasoning tasks (faster than full model)
        if features.has_reasoning_request and complexity < 0.85:
            return GPTVariant.GPT5_CHAT
            
        # Default to mini to avoid full model latency
        return GPTVariant.GPT5_MINI
    
    def _route_for_cost(self, complexity: float, features: PromptFeatures) -> GPTVariant:
        """Optimize for lowest cost while maintaining acceptable quality"""
        # Only use full model for extremely complex tasks
        if complexity > 0.95:
            return GPTVariant.GPT5_FULL
            
        # Use nano for very simple tasks
        if complexity < 0.25:
            return GPTVariant.GPT5_NANO
            
        # Push mini to its limits before upgrading
        if complexity < 0.8:
            return GPTVariant.GPT5_MINI
            
        # Use chat instead of full when possible
        return GPTVariant.GPT5_CHAT
    
    def _route_for_quality(self, complexity: float, features: PromptFeatures) -> GPTVariant:
        """Optimize for best possible output quality"""
        # Use full model for complex or quality-sensitive tasks
        if complexity > 0.8 or features.quality_sensitive:
            return GPTVariant.GPT5_FULL
            
        # Use chat for reasoning tasks
        if features.has_reasoning_request:
            return GPTVariant.GPT5_CHAT
            
        # Use mini for medium complexity
        if complexity > 0.4:
            return GPTVariant.GPT5_MINI
            
        # Even simple tasks get mini for better quality
        return GPTVariant.GPT5_MINI
    
    def calculate_routing_confidence(self, selected_variant: GPTVariant, 
                                   complexity: float, features: PromptFeatures) -> float:
        """Calculate confidence in routing decision (0.0 - 1.0)"""
        confidence = 0.5  # Base confidence
        
        # High confidence for clear routing cases
        if complexity < 0.2 and selected_variant == GPTVariant.GPT5_NANO:
            confidence = 0.9
        elif complexity > 0.9 and selected_variant == GPTVariant.GPT5_FULL:
            confidence = 0.95
        elif features.has_reasoning_request and selected_variant == GPTVariant.GPT5_CHAT:
            confidence = 0.85
        elif 0.3 < complexity < 0.7 and selected_variant == GPTVariant.GPT5_MINI:
            confidence = 0.8
        
        # Lower confidence for edge cases
        elif 0.25 < complexity < 0.35:  # Border between nano and mini
            confidence = 0.6
        elif 0.65 < complexity < 0.75:  # Border between mini and chat/full
            confidence = 0.65
        
        return confidence
    
    def generate_routing_explanation(self, selected_variant: GPTVariant, 
                                   complexity: float, features: PromptFeatures,
                                   preferences: UserPreferences) -> str:
        """Generate human-readable explanation for routing decision"""
        explanations = []
        
        # Complexity explanation
        if complexity < 0.3:
            explanations.append(f"Low complexity task (score: {complexity:.2f})")
        elif complexity < 0.7:
            explanations.append(f"Medium complexity task (score: {complexity:.2f})")
        else:
            explanations.append(f"High complexity task (score: {complexity:.2f})")
        
        # Feature-based explanations
        if features.has_code_request:
            explanations.append("Code generation detected")
        if features.has_reasoning_request:
            explanations.append("Multi-step reasoning required")
        if features.is_technical:
            explanations.append("Technical domain expertise needed")
        if features.quality_sensitive:
            explanations.append("Quality-sensitive task identified")
        
        # Priority explanation
        priority_text = {
            Priority.SPEED: "Speed optimized: selected fastest capable model",
            Priority.COST: "Cost optimized: selected most economical option", 
            Priority.QUALITY: "Quality optimized: selected best performing model"
        }
        explanations.append(priority_text[preferences.priority])
        
        return " • ".join(explanations)
    
    def get_alternative_recommendations(self, selected_variant: GPTVariant,
                                      complexity: float) -> List[Dict[str, Any]]:
        """Get alternative model recommendations with trade-offs"""
        alternatives = []
        
        for variant in GPTVariant:
            if variant != selected_variant:
                specs = self.model_specs[variant]
                selected_specs = self.model_specs[selected_variant]
                
                cost_diff = specs.cost_per_1k_tokens - selected_specs.cost_per_1k_tokens
                latency_diff = specs.avg_latency_ms - selected_specs.avg_latency_ms
                quality_diff = specs.quality_score - selected_specs.quality_score
                
                alternatives.append({
                    "variant": variant.value,
                    "name": specs.name,
                    "cost_difference": f"${cost_diff:.3f}/1K tokens",
                    "latency_difference": f"{latency_diff:+d}ms",
                    "quality_difference": f"{quality_diff:+.1f} points",
                    "recommendation": self._get_alternative_reason(variant, selected_variant, complexity)
                })
        
        return alternatives
    
    def _get_alternative_reason(self, alternative: GPTVariant, selected: GPTVariant, 
                               complexity: float) -> str:
        """Get reason why alternative might be considered"""
        if alternative == GPTVariant.GPT5_NANO:
            return "Consider for maximum speed and cost savings on simple tasks"
        elif alternative == GPTVariant.GPT5_MINI:
            return "Good balance of cost, speed, and quality for most tasks"
        elif alternative == GPTVariant.GPT5_CHAT:
            return "Better for reasoning and step-by-step explanations"
        elif alternative == GPTVariant.GPT5_FULL:
            return "Maximum quality for complex or critical tasks"
        return "Alternative option available"
    
    async def route_with_analysis(self, prompt: str, 
                                user_preferences: Dict[str, Any]) -> RoutingResult:
        """
        Main routing method with full analysis and explanation
        """
        # Parse user preferences
        preferences = UserPreferences(
            priority=Priority(user_preferences.get('priority', 'quality')),
            max_cost=user_preferences.get('max_cost', 0.02),
            max_latency=user_preferences.get('max_latency', 5000),
            speed_weight=user_preferences.get('speed_weight', 0.33),
            cost_weight=user_preferences.get('cost_weight', 0.33),
            quality_weight=user_preferences.get('quality_weight', 0.34)
        )
        
        # Extract features and calculate complexity
        features = self.extract_prompt_features(prompt)
        complexity = self.calculate_complexity_score(prompt, features)
        
        # Route based on priority
        selected_variant = self.route_by_priority(complexity, features, preferences.priority)
        
        # Check constraints
        selected_specs = self.model_specs[selected_variant]
        if (selected_specs.cost_per_1k_tokens > preferences.max_cost or 
            selected_specs.avg_latency_ms > preferences.max_latency):
            # Find alternative that meets constraints
            selected_variant = self._find_constrained_alternative(preferences, complexity, features)
        
        # Calculate metrics
        confidence = self.calculate_routing_confidence(selected_variant, complexity, features)
        explanation = self.generate_routing_explanation(selected_variant, complexity, features, preferences)
        alternatives = self.get_alternative_recommendations(selected_variant, complexity)
        
        # Estimate performance
        estimated_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        estimated_cost = (estimated_tokens / 1000) * selected_specs.cost_per_1k_tokens
        
        return RoutingResult(
            selected_variant=selected_variant,
            complexity_score=complexity,
            confidence=confidence,
            reasoning=explanation,
            estimated_cost=estimated_cost,
            estimated_latency=selected_specs.avg_latency_ms,
            alternatives=alternatives,
            performance_prediction={
                "quality_score": selected_specs.quality_score,
                "cost_efficiency": 10 - (selected_specs.cost_per_1k_tokens * 100),
                "speed_score": 10 - (selected_specs.avg_latency_ms / 200)
            }
        )
    
    def _find_constrained_alternative(self, preferences: UserPreferences, 
                                    complexity: float, features: PromptFeatures) -> GPTVariant:
        """Find alternative model that meets user constraints"""
        # Try variants in order of preference based on priority
        if preferences.priority == Priority.SPEED:
            candidates = [GPTVariant.GPT5_NANO, GPTVariant.GPT5_MINI, GPTVariant.GPT5_CHAT]
        elif preferences.priority == Priority.COST:
            candidates = [GPTVariant.GPT5_NANO, GPTVariant.GPT5_MINI, GPTVariant.GPT5_CHAT]  
        else:  # Quality
            candidates = [GPTVariant.GPT5_MINI, GPTVariant.GPT5_CHAT, GPTVariant.GPT5_NANO]
            
        for variant in candidates:
            specs = self.model_specs[variant]
            if (specs.cost_per_1k_tokens <= preferences.max_cost and 
                specs.avg_latency_ms <= preferences.max_latency):
                return variant
                
        # Fallback to cheapest/fastest option
        return GPTVariant.GPT5_NANO


# Example usage and testing
if __name__ == "__main__":
    router = GPT5VariantRouter()
    
    # Test cases
    test_prompts = [
        {
            "prompt": "What's the weather like today?",
            "preferences": {"priority": "speed"}
        },
        {
            "prompt": "Write a Python function to sort a list of dictionaries by multiple keys",
            "preferences": {"priority": "cost"}
        },
        {
            "prompt": "Think step by step: analyze the economic implications of AI automation on employment markets, considering historical precedents and potential policy responses",
            "preferences": {"priority": "quality"}
        },
        {
            "prompt": "Create a complete full-stack web application for project management with user authentication, real-time collaboration, file uploads, and comprehensive reporting dashboard",
            "preferences": {"priority": "quality"}
        }
    ]
    
    async def test_routing():
        for i, test in enumerate(test_prompts, 1):
            print(f"\n=== Test Case {i} ===")
            print(f"Prompt: {test['prompt'][:100]}...")
            print(f"Preference: {test['preferences']['priority']}")
            
            result = await router.route_with_analysis(test['prompt'], test['preferences'])
            
            print(f"Selected Model: {result.selected_variant.value}")
            print(f"Complexity Score: {result.complexity_score:.2f}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Estimated Cost: ${result.estimated_cost:.4f}")
            print(f"Estimated Latency: {result.estimated_latency}ms")
            print(f"Reasoning: {result.reasoning}")
    
    import asyncio
    asyncio.run(test_routing())