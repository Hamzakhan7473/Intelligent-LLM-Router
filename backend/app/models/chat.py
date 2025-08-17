from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_items=1, description="List of chat messages")
    model: Optional[str] = Field(None, description="Specific model to use (optional for intelligent routing)")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(1000, gt=0, le=4000, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Enable streaming response")
    
    # Advanced routing parameters
    prefer_speed: Optional[bool] = Field(False, description="Prioritize response speed")
    prefer_quality: Optional[bool] = Field(True, description="Prioritize response quality")
    task_type: Optional[Literal["chat", "creative", "analytical", "coding", "summarization", "reasoning"]] = Field(
        "chat", description="Type of task for intelligent routing"
    )
    
    # Provider preferences
    preferred_provider: Optional[Literal["openai", "anthropic", "google", "auto"]] = Field(
        "auto", description="Preferred LLM provider (auto for intelligent selection)"
    )
    
    # Cost constraints
    max_cost_per_request: Optional[float] = Field(None, description="Maximum cost per request in USD")
    
    # Advanced parameters
    context_window_preference: Optional[Literal["small", "medium", "large"]] = Field(
        "medium", description="Preferred context window size"
    )

class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    # Enhanced usage tracking
    estimated_cost: Optional[float] = Field(None, description="Estimated cost in USD")
    cost_per_1k_tokens: Optional[float] = Field(None, description="Cost per 1K tokens for this model")

class RoutingDecision(BaseModel):
    """Detailed routing decision information"""
    selected_model: str
    selected_provider: str
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in routing decision (0-1)")
    alternatives_considered: List[Dict[str, Any]] = Field(default_factory=list)
    routing_factors: Dict[str, Any] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    message: ChatMessage
    model: str
    usage: ChatUsage
    routing_decision: Optional[str] = Field(None, description="Simple explanation of routing decision")
    provider: str = Field(..., description="LLM provider used")
    response_time_ms: int = Field(..., description="Response time in milliseconds")
    
    # Enhanced routing information
    detailed_routing: Optional[RoutingDecision] = Field(None, description="Detailed routing analysis")
    
    # Provider-specific metadata
    provider_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Provider-specific response metadata")

class ModelInfo(BaseModel):
    """Information about available models"""
    model_id: str
    provider: Literal["openai", "anthropic", "google"]
    display_name: str
    description: str
    max_tokens: int
    cost_per_1k_tokens: float
    avg_latency_ms: int
    capabilities: List[str]
    available: bool = True
    
class ProvidersStatus(BaseModel):
    """Status of all providers"""
    openai: Dict[str, Any] = Field(default_factory=dict)
    anthropic: Dict[str, Any] = Field(default_factory=dict) 
    google: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)

class RoutingStats(BaseModel):
    """Routing statistics"""
    total_requests: int = 0
    requests_by_provider: Dict[str, int] = Field(default_factory=dict)
    requests_by_model: Dict[str, int] = Field(default_factory=dict)
    total_cost: float = 0.0
    avg_response_time_ms: float = 0.0
    cost_savings_percentage: float = 0.0

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str
    providers_status: Optional[ProvidersStatus] = None
    routing_stats: Optional[RoutingStats] = None

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    provider: Optional[str] = Field(None, description="Provider that caused the error")
    model: Optional[str] = Field(None, description="Model that caused the error")
    
# Supported models configuration
SUPPORTED_MODELS = {
    # OpenAI Models
    "gpt-3.5-turbo": {
        "provider": "openai",
        "display_name": "GPT-3.5 Turbo",
        "cost_per_1k_tokens": 0.002,
        "max_tokens": 4096,
        "capabilities": ["chat", "coding", "summarization"],
        "avg_latency_ms": 500,
        "available": True
    },
    "gpt-4": {
        "provider": "openai", 
        "display_name": "GPT-4",
        "cost_per_1k_tokens": 0.030,
        "max_tokens": 8192,
        "capabilities": ["chat", "coding", "analytical", "reasoning", "creative"],
        "avg_latency_ms": 1200,
        "available": True
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "display_name": "GPT-4 Turbo", 
        "cost_per_1k_tokens": 0.010,
        "max_tokens": 128000,
        "capabilities": ["chat", "coding", "analytical", "reasoning", "creative", "summarization"],
        "avg_latency_ms": 800,
        "available": True
    },
    
    # Future OpenAI Models (placeholders)
    "gpt-5-mini": {
        "provider": "openai",
        "display_name": "GPT-5 Mini",
        "cost_per_1k_tokens": 0.005,
        "max_tokens": 8192,
        "capabilities": ["chat", "coding", "analytical", "reasoning"],
        "avg_latency_ms": 600,
        "available": False  # Not yet released
    },
    "gpt-5": {
        "provider": "openai",
        "display_name": "GPT-5",
        "cost_per_1k_tokens": 0.040,
        "max_tokens": 32000,
        "capabilities": ["chat", "coding", "analytical", "reasoning", "creative", "summarization"],
        "avg_latency_ms": 1000,
        "available": False  # Not yet released
    },
    
    # Anthropic Models
    "claude-3-haiku-20240307": {
        "provider": "anthropic",
        "display_name": "Claude 3 Haiku",
        "cost_per_1k_tokens": 0.0025,
        "max_tokens": 4096,
        "capabilities": ["chat", "summarization"],
        "avg_latency_ms": 400,
        "available": True
    },
    "claude-3-sonnet-20240229": {
        "provider": "anthropic",
        "display_name": "Claude 3 Sonnet", 
        "cost_per_1k_tokens": 0.008,
        "max_tokens": 4096,
        "capabilities": ["chat", "creative", "analytical", "coding"],
        "avg_latency_ms": 800,
        "available": True
    },
    "claude-3-opus-20240229": {
        "provider": "anthropic",
        "display_name": "Claude 3 Opus",
        "cost_per_1k_tokens": 0.015,
        "max_tokens": 4096, 
        "capabilities": ["chat", "creative", "analytical", "reasoning", "coding"],
        "avg_latency_ms": 1200,
        "available": True
    },
    
    # Google Gemini Models  
    "gemini-1.5-flash": {
        "provider": "google",
        "display_name": "Gemini 1.5 Flash",
        "cost_per_1k_tokens": 0.001,
        "max_tokens": 8192,
        "capabilities": ["chat", "summarization", "coding"],
        "avg_latency_ms": 300,
        "available": True
    },
    "gemini-1.5-pro": {
        "provider": "google",
        "display_name": "Gemini 1.5 Pro",
        "cost_per_1k_tokens": 0.007,
        "max_tokens": 32000,
        "capabilities": ["chat", "creative", "analytical", "reasoning", "coding", "summarization"],
        "avg_latency_ms": 700,
        "available": True
    },
    "gemini-1.0-pro": {
        "provider": "google", 
        "display_name": "Gemini 1.0 Pro",
        "cost_per_1k_tokens": 0.005,
        "max_tokens": 8192,
        "capabilities": ["chat", "analytical", "coding"],
        "avg_latency_ms": 600,
        "available": True
    }
}

# Task type to capability mapping
TASK_CAPABILITIES_MAP = {
    "chat": "chat",
    "creative": "creative", 
    "analytical": "analytical",
    "coding": "coding",
    "summarization": "summarization",
    "reasoning": "reasoning"
}