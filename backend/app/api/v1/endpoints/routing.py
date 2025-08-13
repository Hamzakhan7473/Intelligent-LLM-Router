# backend/app/api/v1/endpoints/routing.py

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
import json
import asyncio
import uuid
import time
from datetime import datetime
import logging

from app.services.llm_service import LLMAPIService
from app.services.database_service import DatabaseService
from app.models.database import get_db
from app.ml.routing.gpt5_router import RoutingResult, GPTVariant
from app.core.config import settings
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response Models
class RoutingRequest(BaseModel):
    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=50000, 
        description="The user prompt to route and process",
        example="Write a Python function to implement bubble sort"
    )
    user_preferences: Dict[str, Any] = Field(
        default={
            "priority": "quality",
            "max_cost": 0.02,
            "max_latency": 5000
        },
        description="User routing preferences",
        example={
            "priority": "cost",
            "max_cost": 0.01,
            "max_latency": 3000
        }
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context like conversation history, system message",
        example={
            "system_message": "You are a helpful coding assistant",
            "conversation_history": []
        }
    )
    stream: bool = Field(
        default=False, 
        description="Whether to stream the response"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for tracking user preferences"
    )

    @validator('user_preferences')
    def validate_user_preferences(cls, v):
        """Validate user preferences"""
        valid_priorities = ["speed", "cost", "quality"]
        if v.get("priority") and v["priority"] not in valid_priorities:
            raise ValueError(f"Priority must be one of: {valid_priorities}")
        
        if v.get("max_cost") and (v["max_cost"] <= 0 or v["max_cost"] > 1.0):
            raise ValueError("max_cost must be between 0 and 1.0")
        
        if v.get("max_latency") and (v["max_latency"] <= 0 or v["max_latency"] > 60000):
            raise ValueError("max_latency must be between 0 and 60000 milliseconds")
        
        return v

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
    include_quality_analysis: bool = Field(
        default=True,
        description="Whether to include quality analysis in comparison"
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
    cost_per_1k_tokens: Optional[float] = None
    last_updated: datetime

# Dependency to get LLM service
async def get_llm_service() -> LLMAPIService:
    """Get LLM service instance"""
    service = LLMAPIService()
    try:
        yield service
    finally:
        await service.client.aclose()

@router.post("/", response_model=RoutingResponse)
async def route_prompt(
    request: RoutingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    request_info: Request = None
):
    """
    🧠 Main routing endpoint - Automatically selects optimal LLM and generates response
    
    This endpoint:
    1. Analyzes the prompt complexity and features
    2. Routes to the best GPT-5 variant based on user preferences
    3. Makes the API call with fallback handling
    4. Returns response with detailed routing explanation
    
    ## Example Usage
    ```json
    {
        "prompt": "Write a Python function to sort a list",
        "user_preferences": {
            "priority": "cost",
            "max_cost": 0.01,
            "max_latency": 3000
        }
    }
    ```
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get or create database service
        db_service = DatabaseService(db)
        
        # Update session preferences
        db_service.create_or_update_session(session_id, request.user_preferences)
        
        # Create LLM service
        async with LLMAPIService() as llm_service:
            # Route and call LLM
            llm_response, routing_result = await llm_service.route_and_call(
                prompt=request.prompt,
                user_preferences=request.user_preferences,
                context=request.context
            )
            
            # Calculate performance metrics
            end_time = time.time()
            total_latency = int((end_time - start_time) * 1000)
            
            # Calculate cost savings
            gpt5_full_cost = (llm_response.tokens_used / 1000) * settings.GPT5_FULL_INPUT_COST
            cost_savings = gpt5_full_cost - llm_response.cost
            savings_percentage = (cost_savings / gpt5_full_cost * 100) if gpt5_full_cost > 0 else 0
            
            # Prepare response
            response_data = RoutingResponse(
                response=llm_response.content,
                routing_decision={
                    "selected_variant": routing_result.selected_variant.value,
                    "complexity_score": routing_result.complexity_score,
                    "confidence": routing_result.confidence,
                    "reasoning": routing_result.reasoning,
                    "alternatives": routing_result.alternatives,
                    "features_detected": routing_result.features_detected
                },
                performance_metrics={
                    "actual_cost": llm_response.cost,
                    "actual_latency_ms": llm_response.latency_ms,
                    "total_request_time_ms": total_latency,
                    "tokens_used": llm_response.tokens_used,
                    "tokens_input": llm_response.tokens_input,
                    "tokens_output": llm_response.tokens_output,
                    "quality_estimate": llm_response.quality_estimate,
                    "estimated_cost": routing_result.estimated_cost,
                    "estimated_latency": routing_result.estimated_latency,
                    "cost_accuracy": abs(llm_response.cost - routing_result.estimated_cost),
                    "latency_accuracy": abs(llm_response.latency_ms - routing_result.estimated_latency),
                    "cost_savings_usd": round(cost_savings, 4),
                    "cost_savings_percent": round(savings_percentage, 1)
                },
                metadata={
                    "request_id": request_id,
                    "session_id": session_id,
                    "provider": llm_response.provider.value,
                    "model": llm_response.model,
                    "timestamp": llm_response.timestamp.isoformat(),
                    "user_agent": request_info.headers.get("user-agent") if request_info else None,
                    "routing_metadata": routing_result.routing_metadata
                }
            )
            
            # Background task for analytics logging
            background_tasks.add_task(
                log_request_analytics,
                db_service,
                request,
                routing_result,
                llm_response,
                request_id,
                session_id,
                total_latency
            )
            
            return response_data
        
    except ValueError as e:
        logger.warning(f"⚠️  Invalid request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Routing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Routing failed: {str(e)}" if settings.DEBUG else "Internal server error"
        )

@router.post("/gpt5-intelligent", response_model=RoutingResponse)
async def intelligent_gpt5_routing(
    request: RoutingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    🎯 GPT-5 specific routing - Only routes between GPT-5 variants
    
    This endpoint focuses specifically on choosing the optimal GPT-5 variant:
    - GPT-5 Nano for simple, fast tasks
    - GPT-5 Mini for balanced performance
    - GPT-5 Chat for reasoning tasks  
    - GPT-5 Full for complex work
    
    Returns detailed cost savings compared to always using GPT-5 Full.
    """
    request_id = str(uuid.uuid4())
    
    try:
        session_id = request.session_id or str(uuid.uuid4())
        db_service = DatabaseService(db)
        
        async with LLMAPIService() as llm_service:
            # Get routing decision
            routing_result = await llm_service.router.route_with_analysis(
                request.prompt, 
                request.user_preferences
            )
            
            # Ensure we're using a GPT-5 variant
            if not routing_result.selected_variant.value.startswith('gpt-5'):
                routing_result.selected_variant = GPTVariant.GPT5_MINI
                logger.info(f"🔄 Forced routing to GPT-5 variant: {routing_result.selected_variant.value}")
            
            # Make API call with GPT-5 specific fallback
            llm_response = await llm_service.call_llm_with_fallback(
                variant=routing_result.selected_variant,
                prompt=request.prompt,
                context=request.context
            )
            
            # Calculate GPT-5 specific savings
            gpt5_full_cost = (llm_response.tokens_used / 1000) * settings.GPT5_FULL_INPUT_COST
            cost_savings = gpt5_full_cost - llm_response.cost
            savings_percentage = (cost_savings / gpt5_full_cost * 100) if gpt5_full_cost > 0 else 0
            
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
                    ],
                    "gpt5_optimization": True
                },
                performance_metrics={
                    "actual_cost": llm_response.cost,
                    "actual_latency_ms": llm_response.latency_ms,
                    "tokens_used": llm_response.tokens_used,
                    "quality_estimate": llm_response.quality_estimate,
                    "cost_savings_vs_gpt5_full": round(cost_savings, 4),
                    "savings_percentage": round(savings_percentage, 1),
                    "gpt5_full_cost": round(gpt5_full_cost, 4)
                },
                metadata={
                    "request_id": request_id,
                    "provider": llm_response.provider.value,
                    "model": llm_response.model,
                    "gpt5_optimization": True,
                    "timestamp": llm_response.timestamp.isoformat()
                }
            )
        
    except Exception as e:
        logger.error(f"❌ GPT-5 routing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"GPT-5 routing failed: {str(e)}" if settings.DEBUG else "Internal server error"
        )

@router.post("/compare", response_model=ComparisonResponse)
async def compare_variants(
    request: ComparisonRequest,
    db: Session = Depends(get_db)
):
    """
    📊 Compare multiple LLM variants on the same prompt
    
    Useful for:
    - Evaluating different models
    - A/B testing responses
    - Understanding cost/quality trade-offs
    
    Returns detailed comparison with recommendations.
    """
    try:
        # Default to all GPT-5 variants if none specified
        variants_to_test = request.variants or [
            "gpt-5-nano", "gpt-5-mini", "gpt-5-chat", "gpt-5"
        ]
        
        # Validate variants
        valid_variants = [variant.value for variant in GPTVariant]
        invalid_variants = [v for v in variants_to_test if v not in valid_variants]
        if invalid_variants:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid variants: {invalid_variants}. Valid options: {valid_variants}"
            )
        
        # Convert to GPTVariant enums
        test_variants = [GPTVariant(variant_str) for variant_str in variants_to_test]
        
        async with LLMAPIService() as llm_service:
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Comparison failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}" if settings.DEBUG else "Internal server error"
        )

@router.get("/status", response_model=List[ModelStatus])
async def get_model_status(db: Session = Depends(get_db)):
    """
    🔍 Get current status of all LLM models
    
    Returns availability, performance metrics, and health status for all supported models.
    """
    try:
        db_service = DatabaseService(db)
        
        async with LLMAPIService() as llm_service:
            # Check availability
            availability = await llm_service.get_model_availability()
            
            # Get performance data for each variant
            status_list = []
            for variant in GPTVariant:
                # Get performance metrics from database
                performance = db_service.get_model_performance(variant.value, days=7)
                
                # Get model specs
                model_specs = llm_service.router.model_specs[variant]
                
                status_list.append(ModelStatus(
                    variant=variant.value,
                    available=availability.get("openai", False),  # Simplified check
                    avg_latency=performance.get("avg_latency"),
                    success_rate=performance.get("success_rate"),
                    cost_per_1k_tokens=model_specs.cost_per_1k_input_tokens,
                    last_updated=datetime.utcnow()
                ))
            
            return status_list
        
    except Exception as e:
        logger.error(f"❌ Status check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}" if settings.DEBUG else "Internal server error"
        )

@router.post("/stream")
async def stream_response(
    request: RoutingRequest,
    db: Session = Depends(get_db)
):
    """
    🌊 Stream LLM response in real-time
    
    Returns Server-Sent Events (SSE) for real-time streaming of the LLM response.
    """
    async def generate_stream():
        try:
            # Send routing info first
            yield f"data: {json.dumps({'type': 'routing_start', 'data': {'status': 'analyzing_prompt'}})}\n\n"
            
            async with LLMAPIService() as llm_service:
                # Get routing decision
                routing_result = await llm_service.router.route_with_analysis(
                    request.prompt, request.user_preferences
                )
                
                # Send routing decision
                yield f"data: {json.dumps({'type': 'routing_decision', 'data': {
                    'variant': routing_result.selected_variant.value,
                    'complexity': routing_result.complexity_score,
                    'confidence': routing_result.confidence,
                    'reasoning': routing_result.reasoning
                }})}\n\n"
                
                # In a real implementation, this would use actual streaming APIs
                # For now, we'll simulate streaming by breaking up the response
                yield f"data: {json.dumps({'type': 'response_start', 'data': {'status': 'generating'}})}\n\n"
                
                # Get full response (in real implementation, this would be streamed)
                llm_response = await llm_service.call_llm_with_fallback(
                    variant=routing_result.selected_variant,
                    prompt=request.prompt,
                    context=request.context
                )
                
                # Simulate streaming by sending chunks
                words = llm_response.content.split()
                for i, word in enumerate(words):
                    chunk_data = {
                        'type': 'content_chunk',
                        'data': {
                            'chunk': word + ' ',
                            'index': i,
                            'total_words': len(words)
                        }
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    await asyncio.sleep(0.05)  # Simulate streaming delay
                
                # Send completion info
                completion_data = {
                    'type': 'response_complete',
                    'data': {
                        'total_tokens': llm_response.tokens_used,
                        'cost': llm_response.cost,
                        'latency_ms': llm_response.latency_ms,
                        'quality': llm_response.quality_estimate
                    }
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
        except Exception as e:
            error_data = {'type': 'error', 'data': {'message': str(e)}}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache", 
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@router.get("/health")
async def health_check():
    """Simple health check for the routing service"""
    try:
        # Basic health check
        return {
            "status": "healthy",
            "service": "intelligent-llm-router",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "features": {
                "gpt5_routing": settings.ENABLE_GPT5_ROUTING,
                "fallback_providers": settings.ENABLE_FALLBACK_PROVIDERS,
                "analytics": settings.ENABLE_ANALYTICS,
                "quality_estimation": settings.ENABLE_QUALITY_ESTIMATION
            }
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

# Helper functions
async def compare_single_variant(service: LLMAPIService, variant: GPTVariant, prompt: str) -> Dict[str, Any]:
    """Compare a single variant against the prompt"""
    start_time = datetime.utcnow()
    
    try:
        response = await service.call_llm_with_fallback(variant, prompt)
        end_time = datetime.utcnow()
        
        return {
            "success": True,
            "response": response.content[:500],  # Truncate for comparison
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            "tokens": response.tokens_used,
            "quality": response.quality_estimate,
            "provider": response.provider.value,
            "model": response.model,
            "timestamp": end_time.isoformat()
        }
    except Exception as e:
        end_time = datetime.utcnow()
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
    
    # Calculate overall score (weighted combination)
    scores = {}
    for variant, data in successful_results.items():
        # Normalize metrics (lower cost and latency is better, higher quality is better)
        cost_score = 1 / (data["cost"] * 1000 + 1)  # Invert and scale
        speed_score = 1 / (data["latency_ms"] / 1000 + 1)  # Invert and scale
        quality_score = data["quality"] / 10  # Scale to 0-1
        
        # Weighted combination
        overall_score = (cost_score * 0.3 + speed_score * 0.3 + quality_score * 0.4)
        scores[variant] = overall_score
    
    best_overall = max(scores.items(), key=lambda x: x[1])[0] if scores else None
    
    return {
        "fastest": {"variant": fastest[0], "latency_ms": fastest[1]["latency_ms"]},
        "cheapest": {"variant": cheapest[0], "cost": cheapest[1]["cost"]},
        "highest_quality": {"variant": highest_quality[0], "quality": highest_quality[1]["quality"]},
        "best_overall": {"variant": best_overall, "score": scores.get(best_overall, 0)},
        "analysis": f"Best overall choice: {best_overall} with balanced performance across cost, speed, and quality"
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
        "cost_range": {"min": min_cost, "max": max_cost},
        "total_range": f"${min_cost:.4f} - ${max_cost:.4f}"
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
        "avg_latency": round(sum(latencies.values()) / len(latencies)),
        "avg_quality": round(sum(qualities.values()) / len(qualities), 1),
        "latency_range": f"{min(latencies.values())}ms - {max(latencies.values())}ms",
        "quality_range": f"{min(qualities.values()):.1f} - {max(qualities.values()):.1f}"
    }

async def log_request_analytics(
    db_service: DatabaseService,
    request: RoutingRequest,
    routing_result: RoutingResult,
    llm_response,
    request_id: str,
    session_id: str,
    total_latency: int
):
    """Background task to log analytics data"""
    try:
        # Calculate cost savings
        gpt5_full_cost = (llm_response.tokens_used / 1000) * settings.GPT5_FULL_INPUT_COST
        cost_savings = gpt5_full_cost - llm_response.cost
        
        # Log to database
        request_data = {
            "request_id": request_id,
            "prompt": request.prompt,
            "category": routing_result.routing_metadata.get("domain", "general"),
            "complexity_score": routing_result.complexity_score,
            "priority": request.user_preferences.get("priority", "quality"),
            "max_cost": request.user_preferences.get("max_cost"),
            "max_latency": request.user_preferences.get("max_latency"),
            "selected_variant": routing_result.selected_variant.value,
            "confidence": routing_result.confidence,
            "reasoning": routing_result.reasoning,
            "actual_model": llm_response.model,
            "actual_provider": llm_response.provider.value,
            "actual_cost": llm_response.cost,
            "actual_latency_ms": llm_response.latency_ms,
            "tokens_input": llm_response.tokens_input,
            "tokens_output": llm_response.tokens_output,
            "tokens_total": llm_response.tokens_used,
            "quality_estimate": llm_response.quality_estimate,
            "success": llm_response.success,
            "error_message": llm_response.error_message,
            "cost_accuracy": abs(llm_response.cost - routing_result.estimated_cost),
            "latency_accuracy": abs(llm_response.latency_ms - routing_result.estimated_latency),
            "session_id": session_id
        }
        
        db_service.log_routing_request(request_data)
        
        # Update session stats
        db_service.update_session_stats(session_id, llm_response.cost, cost_savings)
        
        # Record system metrics
        db_service.record_system_metric("total_requests", 1, "counter")
        db_service.record_system_metric("total_cost", llm_response.cost, "counter", "USD")
        db_service.record_system_metric("total_savings", cost_savings, "counter", "USD")
        
        logger.info(f"✅ Analytics logged for request: {request_id}")
        
    except Exception as e:
        logger.error(f"❌ Failed to log analytics: {str(e)}")