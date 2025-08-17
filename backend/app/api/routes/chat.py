from fastapi import APIRouter, HTTPException, Depends
from app.models.chat import ChatRequest, ChatResponse, ErrorResponse
from app.services.chat_service import ChatService
from app.services.routing_service import RoutingService
import time
from datetime import datetime

router = APIRouter()

# Initialize services
chat_service = ChatService()
routing_service = RoutingService()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint with intelligent LLM routing
    """
    start_time = time.time()
    
    try:
        # Step 1: Intelligent routing decision
        if request.model is None:
            # Use intelligent routing to select best model
            routing_decision = await routing_service.select_optimal_model(
                messages=request.messages,
                task_type=request.task_type,
                prefer_speed=request.prefer_speed,
                prefer_quality=request.prefer_quality
            )
            selected_model = routing_decision["model"]
            routing_explanation = routing_decision["reasoning"]
        else:
            # User specified a model
            selected_model = request.model
            routing_explanation = f"User-specified model: {request.model}"
        
        # Step 2: Generate response using selected model
        response_data = await chat_service.generate_response(
            messages=request.messages,
            model=selected_model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Return structured response
        return ChatResponse(
            message=response_data["message"],
            model=selected_model,
            usage=response_data["usage"],
            routing_decision=routing_explanation,
            provider=response_data["provider"],
            response_time_ms=response_time_ms
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat generation failed: {str(e)}"
        )

@router.get("/models")
async def list_available_models():
    """
    List all available models and their capabilities
    """
    try:
        models = await chat_service.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve models: {str(e)}"
        )

@router.post("/route")
async def get_routing_recommendation(request: ChatRequest):
    """
    Get routing recommendation without generating response
    """
    try:
        routing_decision = await routing_service.select_optimal_model(
            messages=request.messages,
            task_type=request.task_type,
            prefer_speed=request.prefer_speed,
            prefer_quality=request.prefer_quality
        )
        return routing_decision
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Routing analysis failed: {str(e)}"
        )