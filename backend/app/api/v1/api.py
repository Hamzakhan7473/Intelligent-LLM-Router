# backend/app/api/v1/api.py

from fastapi import APIRouter
from app.api.v1.endpoints import routing, models, analytics, evaluation

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    routing.router, 
    prefix="/route", 
    tags=["routing"],
    responses={
        200: {"description": "Successful routing"},
        400: {"description": "Bad request"},
        500: {"description": "Internal server error"}
    }
)

api_router.include_router(
    models.router, 
    prefix="/models", 
    tags=["models"],
    responses={
        200: {"description": "Model operation successful"},
        404: {"description": "Model not found"},
        500: {"description": "Internal server error"}
    }
)

api_router.include_router(
    analytics.router, 
    prefix="/analytics", 
    tags=["analytics"],
    responses={
        200: {"description": "Analytics data retrieved"},
        500: {"description": "Internal server error"}
    }
)

api_router.include_router(
    evaluation.router, 
    prefix="/evaluate", 
    tags=["evaluation"],
    responses={
        200: {"description": "Evaluation completed"},
        400: {"description": "Invalid evaluation request"},
        500: {"description": "Internal server error"}
    }
)