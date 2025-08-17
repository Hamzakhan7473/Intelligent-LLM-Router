from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from app.api.routes import chat
from app.core.config import settings

# Create FastAPI app
app = FastAPI(
    title="Intelligent LLM Router API",
    description="Backend API for intelligent routing between LLM providers",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # Use the new property method
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat.router, prefix="/api", tags=["chat"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Intelligent LLM Router API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "intelligent-llm-router-backend",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )