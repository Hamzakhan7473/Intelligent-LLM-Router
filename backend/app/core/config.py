from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Intelligent LLM Router"
    
    # CORS
    BACKEND_CORS_ORIGINS: str = "http://localhost:3000"
    
    # LLM Provider API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    
    # Model Configuration
    DEFAULT_MODEL: str = "gpt-3.5-turbo"
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.7
    
    # Routing Configuration
    ENABLE_INTELLIGENT_ROUTING: bool = True
    ROUTING_MODEL: str = "gpt-3.5-turbo"
    
    # Performance Settings
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    
    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.BACKEND_CORS_ORIGINS.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # This allows extra fields in .env

settings = Settings()
