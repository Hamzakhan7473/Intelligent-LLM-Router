# backend/app/api/v1/endpoints/routing.py

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import json
import asyncio
from datetime import datetime

from app.services.llm_service import LLMAPIService, LLMResponse
from app.ml.routing.gpt5_router import RoutingResult, GPTVariant

router = APIRouter()

# Request/Response Models
class RoutingRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=50000, description="The user prompt to route and process")
    user_preferences: Dict[str, Any] = Field(
        default={
            "priority": "quality",  # speed, cost, quality
            "max_cost": 0.02,       # Maximum cost per request
            "max_latency": 5000     # Maximum latency in milliseconds
        },
        description="User routing preferences"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context like conversation history, system message"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")

class RoutingResponse(BaseModel):
    response: str = Field(..., description="The LLM response content")
    routing_decision: Dict[str, Any] = Field(..., description="Details about routing decision")
    performance_metrics: Dict[str, Any] = Field(..., description="Actual performance metrics")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")

class ComparisonRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    variants: Optional[List[str]] = Field(
        default=None,
        description="Specific variants to compare (default: all GPT-5 variants)"
    )

class ComparisonResponse(BaseModel):
    results: Dict[str, Dict[str, Any]] = Field(..., description="Results from each variant")
    recommendation: Dict[str, Any] = Field(..., description="Recommended variant based on analysis")
    cost_comparison: Dict[str, Any] = Field(..., description="Cost analysis")
    performance_comparison: Dict[str, Any] = Field(..., description="Performance analysis")

class ModelStatus(BaseModel):
    variant: str
    available: bool
    avg_latency: Optional[float] = None
    success_rate: Optional[float] = None
    last_updated: datetime

# Dependency to get LLM service
async def get_llm_service() -> LLMAPIService:
    service = LLMAPIService()
    try:
        yield service
    finally:
        await service.client.aclose()

@router.post("/", response_model=RoutingResponse)
async def route_prompt(
    request: RoutingRequest,
    background_tasks: BackgroundTasks,
    llm_service: LLMAPIService = Depends(get_llm_service)
):
    """
    🧠 Main routing endpoint - Automatically selects optimal LLM and generates response
    
    This endpoint:
    1. Analyzes the prompt complexity and features
    2. Routes to the best GPT-5 variant based on user preferences
    3. Makes the API call with fallback handling
    4. Returns response with detailed routing explanation
    """
    try:
        # Validate preferences
        valid_priorities = ["speed", "cost", "quality"]
        if request.user_preferences.get("priority") not in valid_priorities:
            raise HTTPException(
                status_code=400, 
                detail=f"Priority must be one of: {valid_priorities}"
            )
        
        # Route and call LLM
        llm_response, routing_result = await llm_service.route_and_call(
            prompt=request.prompt,
            user_preferences=request.user_preferences,
            context=request.context
        )
        
        # Prepare response
        response_data = RoutingResponse(
            response=llm_response.content,
            routing_decision={
                "selected_variant": routing_result.selected_variant.value,
                "complexity_score": routing_result.complexity_score,
                "confidence": routing_result.confidence,
                "reasoning": routing_result.reasoning,
                "alternatives": routing_result.alternatives
            },
            performance_metrics={
                "actual_cost": llm_response.cost,
                "actual_latency_ms": llm_response.latency_ms,
                "tokens_used": llm_response.tokens_used,
                "quality_estimate": llm_response.quality_estimate,
                "estimated_cost": routing_result.estimated_cost,
                "estimated_latency": routing_result.estimated_latency,
                "cost_accuracy": abs(llm_response.cost - routing_result.estimated_cost),
                "latency_accuracy": abs(llm_response.latency_ms - routing_result.estimated_latency)
            },
            metadata={
                "provider": llm_response.provider.value,
                "model": llm_response.model,
                "timestamp": llm_response.timestamp.isoformat(),
                "request_id": f"req_{int(datetime.now().timestamp())}"
            }
        )
        
        # Background task for analytics (non-blocking)
        background_tasks.add_task(
            log_request_analytics,
            request.prompt,
            routing_result,
            llm_response
        )
        
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Routing failed: {str(e)}"
        )

@router.post("/gpt5-intelligent", response_model=RoutingResponse)
async def intelligent_gpt5_routing(
    request: RoutingRequest,
    llm_service: LLMAPIService = Depends(get_llm_service)
):
    """
    🎯 GPT-5 specific routing - Only routes between GPT-5 variants
    
    This endpoint focuses specifically on choosing the optimal GPT-5 variant:
    - GPT-5 Nano for simple, fast tasks
    - GPT-5 Mini for balanced performance
    - GPT-5 Chat for reasoning tasks  
    - GPT-5 Full for complex work
    """
    try:
        # Force routing to only consider GPT-5 variants
        routing_result = await llm_service.router.route_with_analysis(
            request.prompt, 
            request.user_preferences
        )
        
        # Ensure we're using a GPT-5 variant
        if not routing_result.selected_variant.value.startswith('gpt-5'):
            # Force to GPT-5 Mini as reasonable default
            routing_result.selected_variant = GPTVariant.GPT5_MINI
        
        # Make API call with GPT-5 specific fallback
        llm_response = await llm_service.call_llm_with_fallback(
            variant=routing_result.selected_variant,
            prompt=request.prompt,
            context=request.context
        )
        
        return RoutingResponse(
            response=llm_response.content,
            routing_decision={
                "selected_variant": routing_result.selected_variant.value,
                "complexity_score": routing_result.complexity_score,
                "confidence": routing_result.confidence,
                "reasoning": routing_result.reasoning + " (GPT-5 variants only)",
                "alternatives": [
                    alt for alt in routing_result.alternatives 
                    if alt["variant"].startswith('gpt-5')
                ]
            },
            performance_metrics={
                "actual_cost": llm_response.cost,
                "actual_latency_ms": llm_response.latency_ms,
                "tokens_used": llm_response.tokens_used,
                "quality_estimate": llm_response.quality_estimate,
                "cost_savings_vs_gpt5_full": calculate_gpt5_savings(
                    routing_result.selected_variant, llm_response.tokens_used
                )
            },
            metadata={
                "provider": llm_response.provider.value,
                "model": llm_response.model,
                "gpt5_optimization": True,
                "timestamp": llm_response.timestamp.isoformat()
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"GPT-5 routing failed: {str(e)}"
        )

@router.post("/compare", response_model=ComparisonResponse)
async def compare_variants(
    request: ComparisonRequest,
    llm_service: LLMAPIService = Depends(get_llm_service)
):
    """
    📊 Compare multiple LLM variants on the same prompt
    
    Useful for:
    - Evaluating different models
    - A/B testing responses
    - Understanding cost/quality trade-offs
    """
    try:
        # Default to all GPT-5 variants if none specified
        variants_to_test = request.variants or [
            "gpt-5-nano", "gpt-5-mini", "gpt-5-chat", "gpt-5"
        ]
        
        # Convert to GPTVariant enums
        test_variants = []
        for variant_str in variants_to_test:
            try:
                variant = GPTVariant(variant_str)
                test_variants.append(variant)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid variant: {variant_str}"
                )
        
        # Run comparisons in parallel
        comparison_tasks = [
            compare_single_variant(llm_service, variant, request.prompt)
            for variant in test_variants
        ]
        
        results = await asyncio.gather(*comparison_tasks, return_exceptions=True)
        
        # Process results
        comparison_results = {}
        for variant, result in zip(test_variants, results):
            if isinstance(result, Exception):
                comparison_results[variant.value] = {
                    "error": str(result),
                    "success": False
                }
            else:
                comparison_results[variant.value] = result
        
        # Analyze results and make recommendation
        recommendation = analyze_comparison_results(comparison_results)
        cost_comparison = generate_cost_comparison(comparison_results)
        performance_comparison = generate_performance_comparison(comparison_results)
        
        return ComparisonResponse(
            results=comparison_results,
            recommendation=recommendation,
            cost_comparison=cost_comparison,
            performance_comparison=performance_comparison
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )

@router.get("/status", response_model=List[ModelStatus])
async def get_model_status(llm_service: LLMAPIService = Depends(get_llm_service)):
    """
    🔍 Get current status of all LLM models
    
    Returns availability, performance metrics, and health status
    """
    try:
        # Check availability
        availability = await llm_service.get_model_availability()
        
        # Mock performance data (in production, this would come from database)
        status_list = []
        for variant in GPTVariant:
            status_list.append(ModelStatus(
                variant=variant.value,
                available=availability.get("openai", False),  # Simplified
                avg_latency=llm_service.model_specs[variant].avg_latency_ms,
                success_rate=0.98 if availability.get("openai", False) else 0.0,
                last_updated=datetime.now()
            ))
        
        return status_list
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )

@router.post("/stream")
async def stream_response(
    request: RoutingRequest,
    llm_service: LLMAPIService = Depends(get_llm_service)
):
    """
    🌊 Stream LLM response in real-time
    
    Returns Server-Sent Events (SSE) for real-time streaming
    """
    async def generate_stream():
        try:
            # Get routing decision first
            routing_result = await llm_service.router.route_with_analysis(
                request.prompt, request.user_preferences
            )
            
            # Send routing info
            yield f"data: {json.dumps({'type': 'routing', 'data': {
                'variant': routing_result.selected_variant.value,
                'complexity': routing_result.complexity_score,
                'reasoning': routing_result.reasoning
            }})}\n\n"
            
            # Mock streaming (in production, this would use actual streaming APIs)
            response_text = "This is a streaming response that would come from the LLM API..."
            words = response_text.split()
            
            for i, word in enumerate(words):
                chunk = {
                    'type': 'content',
                    'data': {'chunk': word + ' ', 'index': i}
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.1)  # Simulate streaming delay
            
            # Send completion
            yield f"data: {json.dumps({'type': 'done', 'data': {'total_tokens': len(words)}})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': str(e)}})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# Helper functions
async def compare_single_variant(service: LLMAPIService, variant: GPTVariant, prompt: str) -> Dict[str, Any]:
    """Compare a single variant against the prompt"""
    start_time = datetime.now()
    
    try:
        response = await service.call_llm_with_fallback(variant, prompt)
        end_time = datetime.now()
        
        return {
            "success": True,
            "response": response.content,
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            "tokens": response.tokens_used,
            "quality": response.quality_estimate,
            "provider": response.provider.value,
            "model": response.model,
            "timestamp": end_time.isoformat()
        }
    except Exception as e:
        end_time = datetime.now()
        return {
            "success": False,
            "error": str(e),
            "latency_ms": int((end_time - start_time).total_seconds() * 1000),
            "timestamp": end_time.isoformat()
        }

def analyze_comparison_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze comparison results and make recommendation"""
    successful_results = {k: v for k, v in results.items() if v.get("success", False)}
    
    if not successful_results:
        return {"error": "No successful results to analyze"}
    
    # Find winners in each category
    fastest = min(successful_results.items(), key=lambda x: x[1]["latency_ms"])
    cheapest = min(successful_results.items(), key=lambda x: x[1]["cost"])
    highest_quality = max(successful_results.items(), key=lambda x: x[1]["quality"])
    
    return {
        "fastest": {"variant": fastest[0], "latency_ms": fastest[1]["latency_ms"]},
        "cheapest": {"variant": cheapest[0], "cost": cheapest[1]["cost"]},
        "highest_quality": {"variant": highest_quality[0], "quality": highest_quality[1]["quality"]},
        "overall_recommendation": fastest[0] if len(successful_results) > 1 else list(successful_results.keys())[0]
    }

def generate_cost_comparison(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate cost comparison analysis"""
    successful_results = {k: v for k, v in results.items() if v.get("success", False)}
    
    if not successful_results:
        return {"error": "No successful results for cost comparison"}
    
    costs = {k: v["cost"] for k, v in successful_results.items()}
    min_cost = min(costs.values())
    max_cost = max(costs.values())
    
    return {
        "costs": costs,
        "savings_vs_most_expensive": {
            variant: f"{((max_cost - cost) / max_cost) * 100:.1f}%"
            for variant, cost in costs.items()
        },
        "cost_range": {"min": min_cost, "max": max_cost}
    }

def generate_performance_comparison(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate performance comparison analysis"""
    successful_results = {k: v for k, v in results.items() if v.get("success", False)}
    
    if not successful_results:
        return {"error": "No successful results for performance comparison"}
    
    latencies = {k: v["latency_ms"] for k, v in successful_results.items()}
    qualities = {k: v["quality"] for k, v in successful_results.items()}
    
    return {
        "latencies": latencies,
        "qualities": qualities,
        "avg_latency": sum(latencies.values()) / len(latencies),
        "avg_quality": sum(qualities.values()) / len(qualities)
    }

def calculate_gpt5_savings(variant: GPTVariant, tokens_used: int) -> float:
    """Calculate savings compared to always using GPT-5 Full"""
    full_cost = (tokens_used / 1000) * 0.015  # GPT-5 Full pricing
    
    variant_costs = {
        GPTVariant.GPT5_NANO: 0.003,
        GPTVariant.GPT5_MINI: 0.008,
        GPTVariant.GPT5_CHAT: 0.012,
        GPTVariant.GPT5_FULL: 0.015
    }
    
    variant_cost = (tokens_used / 1000) * variant_costs[variant]
    savings = full_cost - variant_cost
    
    return round(savings, 4)

async def log_request_analytics(prompt: str, routing_result: RoutingResult, llm_response: LLMResponse):
    """Background task to log analytics data"""
    analytics_data = {
        "timestamp": datetime.now().isoformat(),
        "prompt_length": len(prompt),
        "selected_variant": routing_result.selected_variant.value,
        "complexity": routing_result.complexity_score,
        "confidence": routing_result.confidence,
        "cost": llm_response.cost,
        "latency": llm_response.latency_ms,
        "tokens": llm_response.tokens_used,
        "quality": llm_response.quality_estimate
    }
    
    # In production, this would go to a database or analytics service
    print(f"📊 Analytics: {json.dumps(analytics_data)}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Simple health check for the routing service"""
    return {
        "status": "healthy",
        "service": "intelligent-llm-router",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }