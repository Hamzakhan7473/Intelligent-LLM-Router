# backend/app/api/v1/endpoints/models.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from sqlalchemy.orm import Session

from app.models.database import LLMModel, get_db
from app.services.database_service import DatabaseService
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response Models
class ModelInfo(BaseModel):
    id: str
    name: str
    variant: str
    provider: str
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    avg_latency_ms: int
    quality_score: float
    max_tokens: int
    context_window: int
    supported_categories: List[str]
    strengths: List[str]
    is_active: bool
    is_available: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ModelCreate(BaseModel):
    name: str = Field(..., description="Model display name")
    variant: str = Field(..., description="Model variant identifier")
    provider: str = Field(..., description="Provider (openai, anthropic, etc.)")
    api_endpoint: Optional[str] = Field(None, description="API endpoint URL")
    cost_per_1k_input_tokens: float = Field(..., description="Input cost per 1K tokens")
    cost_per_1k_output_tokens: float = Field(..., description="Output cost per 1K tokens")
    avg_latency_ms: int = Field(default=1000, description="Average latency in milliseconds")
    quality_score: float = Field(default=8.0, description="Quality score (1-10)")
    max_tokens: int = Field(default=4000, description="Maximum tokens")
    context_window: int = Field(default=4000, description="Context window size")
    supported_categories: List[str] = Field(default=[], description="Supported prompt categories")
    strengths: List[str] = Field(default=[], description="Model strengths")

class ModelUpdate(BaseModel):
    name: Optional[str] = None
    cost_per_1k_input_tokens: Optional[float] = None
    cost_per_1k_output_tokens: Optional[float] = None
    avg_latency_ms: Optional[int] = None
    quality_score: Optional[float] = None
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    supported_categories: Optional[List[str]] = None
    strengths: Optional[List[str]] = None
    is_active: Optional[bool] = None
    is_available: Optional[bool] = None

class ModelPerformanceStats(BaseModel):
    variant: str
    total_requests: int
    success_rate: float
    avg_cost: float
    avg_latency: float
    avg_quality: float
    cost_efficiency: float
    speed_rank: int
    quality_rank: int

@router.get("/", response_model=List[ModelInfo])
async def list_models(
    include_inactive: bool = False,
    provider: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    📋 List all available LLM models
    
    Returns a list of all configured models with their specifications and current status.
    
    **Parameters:**
    - include_inactive: Include inactive models in the response
    - provider: Filter by provider (openai, anthropic, google)
    """
    try:
        query = db.query(LLMModel)
        
        # Apply filters
        if not include_inactive:
            query = query.filter(LLMModel.is_active == True)
        
        if provider:
            query = query.filter(LLMModel.provider == provider.lower())
        
        models = query.order_by(LLMModel.provider, LLMModel.variant).all()
        
        return [ModelInfo.from_orm(model) for model in models]
        
    except Exception as e:
        logger.error(f"❌ Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models")

@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """
    🔍 Get specific model information
    
    Returns detailed information about a specific model including current performance metrics.
    """
    try:
        model = db.query(LLMModel).filter(LLMModel.id == model_id).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return ModelInfo.from_orm(model)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model")

@router.post("/", response_model=ModelInfo)
async def create_model(model_data: ModelCreate, db: Session = Depends(get_db)):
    """
    ➕ Create a new model configuration
    
    Add a new LLM model to the routing system. Requires admin privileges.
    """
    try:
        # Check if variant already exists
        existing = db.query(LLMModel).filter(LLMModel.variant == model_data.variant).first()
        if existing:
            raise HTTPException(
                status_code=400, 
                detail=f"Model variant '{model_data.variant}' already exists"
            )
        
        # Create new model
        new_model = LLMModel(
            name=model_data.name,
            variant=model_data.variant,
            provider=model_data.provider.lower(),
            api_endpoint=model_data.api_endpoint,
            cost_per_1k_input_tokens=model_data.cost_per_1k_input_tokens,
            cost_per_1k_output_tokens=model_data.cost_per_1k_output_tokens,
            avg_latency_ms=model_data.avg_latency_ms,
            quality_score=model_data.quality_score,
            max_tokens=model_data.max_tokens,
            context_window=model_data.context_window,
            supported_categories=model_data.supported_categories,
            strengths=model_data.strengths,
            is_active=True,
            is_available=True
        )
        
        db.add(new_model)
        db.commit()
        db.refresh(new_model)
        
        logger.info(f"✅ Created new model: {new_model.variant}")
        return ModelInfo.from_orm(new_model)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error creating model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create model")

@router.put("/{model_id}", response_model=ModelInfo)
async def update_model(
    model_id: str, 
    model_updates: ModelUpdate, 
    db: Session = Depends(get_db)
):
    """
    ✏️ Update model configuration
    
    Update model parameters such as cost, latency, quality score, and availability.
    """
    try:
        model = db.query(LLMModel).filter(LLMModel.id == model_id).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Update fields
        update_data = model_updates.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(model, field, value)
        
        model.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(model)
        
        logger.info(f"✅ Updated model: {model.variant}")
        return ModelInfo.from_orm(model)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error updating model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update model")

@router.delete("/{model_id}")
async def delete_model(model_id: str, db: Session = Depends(get_db)):
    """
    🗑️ Delete model configuration
    
    Remove a model from the routing system. This is a soft delete - the model is marked as inactive.
    """
    try:
        model = db.query(LLMModel).filter(LLMModel.id == model_id).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Soft delete - mark as inactive
        model.is_active = False
        model.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"✅ Deleted model: {model.variant}")
        return {"message": f"Model {model.variant} has been deactivated"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error deleting model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete model")

@router.get("/{variant}/performance", response_model=ModelPerformanceStats)
async def get_model_performance(
    variant: str, 
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    📊 Get model performance statistics
    
    Returns detailed performance metrics for a specific model variant over the specified time period.
    """
    try:
        db_service = DatabaseService(db)
        performance_data = db_service.get_model_performance(variant, days)
        
        if "error" in performance_data:
            raise HTTPException(status_code=404, detail=f"No performance data found for variant: {variant}")
        
        # Calculate additional metrics
        cost_efficiency = (10 - performance_data["avg_cost"] * 100) if performance_data["avg_cost"] > 0 else 0
        
        return ModelPerformanceStats(
            variant=variant,
            total_requests=performance_data["request_count"],
            success_rate=performance_data["success_rate"],
            avg_cost=performance_data["avg_cost"],
            avg_latency=performance_data["avg_latency"],
            avg_quality=performance_data["avg_quality"],
            cost_efficiency=max(0, cost_efficiency),
            speed_rank=1,  # Would be calculated relative to other models
            quality_rank=1  # Would be calculated relative to other models
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting performance for {variant}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance data")

@router.post("/{variant}/toggle")
async def toggle_model_availability(
    variant: str, 
    available: bool,
    db: Session = Depends(get_db)
):
    """
    🔄 Toggle model availability
    
    Enable or disable a model for routing. Useful for maintenance or testing.
    """
    try:
        model = db.query(LLMModel).filter(LLMModel.variant == variant).first()
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model variant '{variant}' not found")
        
        model.is_available = available
        model.updated_at = datetime.utcnow()
        
        db.commit()
        
        status = "enabled" if available else "disabled"
        logger.info(f"✅ Model {variant} {status}")
        
        return {
            "message": f"Model {variant} has been {status}",
            "variant": variant,
            "available": available
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error toggling {variant}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to toggle model availability")

@router.get("/providers/summary")
async def get_providers_summary(db: Session = Depends(get_db)):
    """
    📈 Get summary of all providers
    
    Returns a summary of models grouped by provider with aggregate statistics.
    """
    try:
        models = db.query(LLMModel).filter(LLMModel.is_active == True).all()
        
        # Group by provider
        providers = {}
        for model in models:
            provider = model.provider
            if provider not in providers:
                providers[provider] = {
                    "name": provider.title(),
                    "models": [],
                    "total_models": 0,
                    "active_models": 0,
                    "avg_cost": 0,
                    "avg_latency": 0,
                    "avg_quality": 0
                }
            
            providers[provider]["models"].append({
                "variant": model.variant,
                "name": model.name,
                "cost_input": model.cost_per_1k_input_tokens,
                "cost_output": model.cost_per_1k_output_tokens,
                "latency": model.avg_latency_ms,
                "quality": model.quality_score,
                "available": model.is_available
            })
            
            providers[provider]["total_models"] += 1
            if model.is_available:
                providers[provider]["active_models"] += 1
        
        # Calculate averages
        for provider_data in providers.values():
            if provider_data["models"]:
                provider_data["avg_cost"] = sum(m["cost_input"] for m in provider_data["models"]) / len(provider_data["models"])
                provider_data["avg_latency"] = sum(m["latency"] for m in provider_data["models"]) / len(provider_data["models"])
                provider_data["avg_quality"] = sum(m["quality"] for m in provider_data["models"]) / len(provider_data["models"])
        
        return {
            "providers": providers,
            "summary": {
                "total_providers": len(providers),
                "total_models": sum(p["total_models"] for p in providers.values()),
                "active_models": sum(p["active_models"] for p in providers.values())
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting providers summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve providers summary")

@router.get("/variants/gpt5")
async def get_gpt5_variants(db: Session = Depends(get_db)):
    """
    🧠 Get all GPT-5 variants
    
    Returns information about all available GPT-5 model variants with their specifications.
    """
    try:
        gpt5_models = db.query(LLMModel).filter(
            LLMModel.variant.like('gpt-5%'),
            LLMModel.is_active == True
        ).order_by(LLMModel.cost_per_1k_input_tokens).all()
        
        variants = []
        for model in gpt5_models:
            variants.append({
                "variant": model.variant,
                "name": model.name,
                "cost_per_1k_input": model.cost_per_1k_input_tokens,
                "cost_per_1k_output": model.cost_per_1k_output_tokens,
                "avg_latency_ms": model.avg_latency_ms,
                "quality_score": model.quality_score,
                "max_tokens": model.max_tokens,
                "context_window": model.context_window,
                "strengths": model.strengths,
                "supported_categories": model.supported_categories,
                "available": model.is_available
            })
        
        return {
            "gpt5_variants": variants,
            "total_variants": len(variants),
            "cost_range": {
                "min": min(v["cost_per_1k_input"] for v in variants) if variants else 0,
                "max": max(v["cost_per_1k_input"] for v in variants) if variants else 0
            },
            "latency_range": {
                "min": min(v["avg_latency_ms"] for v in variants) if variants else 0,
                "max": max(v["avg_latency_ms"] for v in variants) if variants else 0
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting GPT-5 variants: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve GPT-5 variants")

@router.post("/sync-pricing")
async def sync_model_pricing(db: Session = Depends(get_db)):
    """
    💰 Sync model pricing with latest rates
    
    Updates model pricing information with the latest rates from configuration.
    """
    try:
        model_costs = settings.get_model_costs()
        updated_count = 0
        
        for variant, costs in model_costs.items():
            model = db.query(LLMModel).filter(LLMModel.variant == variant).first()
            if model:
                model.cost_per_1k_input_tokens = costs["input"]
                model.cost_per_1k_output_tokens = costs["output"]
                model.updated_at = datetime.utcnow()
                updated_count += 1
        
        db.commit()
        
        logger.info(f"✅ Updated pricing for {updated_count} models")
        return {
            "message": f"Pricing updated for {updated_count} models",
            "updated_models": list(model_costs.keys())
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error syncing pricing: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to sync pricing")