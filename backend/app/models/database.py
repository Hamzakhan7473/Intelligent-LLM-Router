# backend/app/models/database.py

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from app.core.config import settings

# Database setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class LLMModel(Base):
    __tablename__ = "llm_models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, index=True)
    variant = Column(String(50), nullable=False)  # gpt-5, gpt-5-mini, etc.
    provider = Column(String(50), nullable=False)  # openai, anthropic, etc.
    api_endpoint = Column(Text)
    cost_per_1k_input_tokens = Column(Float, nullable=False)
    cost_per_1k_output_tokens = Column(Float, nullable=False)
    avg_latency_ms = Column(Integer, default=1000)
    quality_score = Column(Float, default=8.0)  # 1-10 scale
    max_tokens = Column(Integer, default=4000)
    context_window = Column(Integer, default=4000)
    supported_categories = Column(JSON)  # List of supported prompt categories
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RoutingRequest(Base):
    __tablename__ = "routing_requests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), index=True)  # Anonymous or user identifier
    session_id = Column(String(100), index=True)
    prompt_hash = Column(String(100), index=True)  # Anonymized prompt identifier
    prompt_length = Column(Integer)
    prompt_category = Column(String(50))  # classified category
    complexity_score = Column(Float)  # 0.0-1.0
    
    # User preferences
    user_priority = Column(String(20))  # speed, cost, quality
    max_cost_preference = Column(Float)
    max_latency_preference = Column(Integer)
    
    # Routing decision
    selected_model_id = Column(UUID(as_uuid=True))
    selected_variant = Column(String(50))
    routing_confidence = Column(Float)
    routing_reasoning = Column(Text)
    
    # Actual performance
    actual_model = Column(String(100))
    actual_provider = Column(String(50))
    actual_cost = Column(Float)
    actual_latency_ms = Column(Integer)
    tokens_input = Column(Integer)
    tokens_output = Column(Integer)
    tokens_total = Column(Integer)
    
    # Quality metrics
    quality_estimate = Column(Float)
    user_rating = Column(Integer)  # 1-5 star rating (optional)
    
    # Status
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)

class ModelPerformance(Base):
    __tablename__ = "model_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), index=True)
    variant = Column(String(50), index=True)
    category = Column(String(50), index=True)
    
    # Performance metrics (aggregated daily)
    date = Column(DateTime, index=True)
    request_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    total_latency_ms = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    
    # Averages
    avg_latency_ms = Column(Float)
    avg_cost_per_request = Column(Float)
    avg_quality_score = Column(Float)
    success_rate = Column(Float)
    
    # Last updated
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(100), unique=True, index=True)
    user_id = Column(String(100), index=True)
    
    # Session preferences
    default_priority = Column(String(20), default="quality")
    default_max_cost = Column(Float, default=0.02)
    default_max_latency = Column(Integer, default=5000)
    
    # Session stats
    total_requests = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    total_savings = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

class SystemMetrics(Base):
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), index=True)
    metric_value = Column(Float)
    metric_type = Column(String(50))  # counter, gauge, histogram
    tags = Column(JSON)  # Additional metadata
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

# Database dependency
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables
def create_tables():
    Base.metadata.create_all(bind=engine)

# Database initialization with sample data
def init_db():
    """Initialize database with sample LLM models"""
    db = SessionLocal()
    
    try:
        # Check if models already exist
        if db.query(LLMModel).first():
            return
        
        # Add GPT-5 variants
        models = [
            LLMModel(
                name="GPT-5 Nano",
                variant="gpt-5-nano",
                provider="openai",
                api_endpoint="https://api.openai.com/v1/chat/completions",
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.004,
                avg_latency_ms=400,
                quality_score=7.9,
                max_tokens=16000,
                context_window=16000,
                supported_categories=["simple_qa", "general_chat", "basic_writing"]
            ),
            LLMModel(
                name="GPT-5 Mini",
                variant="gpt-5-mini", 
                provider="openai",
                api_endpoint="https://api.openai.com/v1/chat/completions",
                cost_per_1k_input_tokens=0.008,
                cost_per_1k_output_tokens=0.010,
                avg_latency_ms=800,
                quality_score=8.8,
                max_tokens=64000,
                context_window=64000,
                supported_categories=["coding", "analysis", "summarization", "translation"]
            ),
            LLMModel(
                name="GPT-5 Chat",
                variant="gpt-5-chat",
                provider="openai", 
                api_endpoint="https://api.openai.com/v1/chat/completions",
                cost_per_1k_input_tokens=0.012,
                cost_per_1k_output_tokens=0.016,
                avg_latency_ms=1000,
                quality_score=9.2,
                max_tokens=128000,
                context_window=128000,
                supported_categories=["reasoning", "explanations", "step_by_step", "analysis"]
            ),
            LLMModel(
                name="GPT-5",
                variant="gpt-5",
                provider="openai",
                api_endpoint="https://api.openai.com/v1/chat/completions", 
                cost_per_1k_input_tokens=0.015,
                cost_per_1k_output_tokens=0.020,
                avg_latency_ms=1200,
                quality_score=9.5,
                max_tokens=128000,
                context_window=128000,
                supported_categories=["complex_reasoning", "app_generation", "research", "creative_writing"]
            ),
            # Fallback models
            LLMModel(
                name="GPT-4o",
                variant="gpt-4o",
                provider="openai",
                api_endpoint="https://api.openai.com/v1/chat/completions",
                cost_per_1k_input_tokens=0.005,
                cost_per_1k_output_tokens=0.015,
                avg_latency_ms=1500,
                quality_score=9.0,
                max_tokens=128000,
                context_window=128000,
                supported_categories=["general", "coding", "analysis"]
            ),
            LLMModel(
                name="Claude 3.5 Sonnet",
                variant="claude-3-5-sonnet-20241022",
                provider="anthropic",
                api_endpoint="https://api.anthropic.com/v1/messages",
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015,
                avg_latency_ms=1100,
                quality_score=9.1,
                max_tokens=200000,
                context_window=200000,
                supported_categories=["reasoning", "coding", "analysis", "creative_writing"]
            )
        ]
        
        for model in models:
            db.add(model)
        
        db.commit()
        print("✅ Database initialized with sample models")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error initializing database: {e}")
    finally:
        db.close()

# backend/app/services/database_service.py

from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from app.models.database import (
    RoutingRequest, LLMModel, ModelPerformance, 
    UserSession, SystemMetrics, get_db
)

class DatabaseService:
    """Service for database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def log_routing_request(self, request_data: Dict[str, Any]) -> RoutingRequest:
        """Log a routing request to the database"""
        routing_request = RoutingRequest(
            user_id=request_data.get('user_id', 'anonymous'),
            session_id=request_data.get('session_id'),
            prompt_hash=str(hash(request_data['prompt']) % 100000),
            prompt_length=len(request_data['prompt']),
            prompt_category=request_data.get('category'),
            complexity_score=request_data.get('complexity_score'),
            user_priority=request_data.get('priority', 'quality'),
            max_cost_preference=request_data.get('max_cost'),
            max_latency_preference=request_data.get('max_latency'),
            selected_variant=request_data.get('selected_variant'),
            routing_confidence=request_data.get('confidence'),
            routing_reasoning=request_data.get('reasoning'),
            actual_model=request_data.get('actual_model'),
            actual_provider=request_data.get('actual_provider'),
            actual_cost=request_data.get('actual_cost'),
            actual_latency_ms=request_data.get('actual_latency_ms'),
            tokens_input=request_data.get('tokens_input'),
            tokens_output=request_data.get('tokens_output'),
            tokens_total=request_data.get('tokens_total'),
            quality_estimate=request_data.get('quality_estimate'),
            success=request_data.get('success', True),
            error_message=request_data.get('error_message'),
            completed_at=datetime.utcnow()
        )
        
        self.db.add(routing_request)
        self.db.commit()
        self.db.refresh(routing_request)
        
        return routing_request
    
    def get_analytics_data(self, days: int = 7) -> Dict[str, Any]:
        """Get analytics data for the dashboard"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Total requests
        total_requests = self.db.query(RoutingRequest).filter(
            RoutingRequest.created_at >= start_date
        ).count()
        
        # Total cost
        total_cost = self.db.query(func.sum(RoutingRequest.actual_cost)).filter(
            RoutingRequest.created_at >= start_date
        ).scalar() or 0.0
        
        # Average latency
        avg_latency = self.db.query(func.avg(RoutingRequest.actual_latency_ms)).filter(
            RoutingRequest.created_at >= start_date
        ).scalar() or 0
        
        # Variant distribution
        variant_stats = self.db.query(
            RoutingRequest.selected_variant,
            func.count(RoutingRequest.id).label('count')
        ).filter(
            RoutingRequest.created_at >= start_date
        ).group_by(RoutingRequest.selected_variant).all()
        
        total_for_percentage = sum(stat.count for stat in variant_stats)
        variant_distribution = {
            stat.selected_variant: round((stat.count / total_for_percentage) * 100, 1)
            for stat in variant_stats
        } if total_for_percentage > 0 else {}
        
        # Daily stats
        daily_stats = self.db.query(
            func.date(RoutingRequest.created_at).label('date'),
            func.count(RoutingRequest.id).label('requests'),
            func.sum(RoutingRequest.actual_cost).label('cost'),
            func.avg(RoutingRequest.actual_latency_ms).label('avg_latency')
        ).filter(
            RoutingRequest.created_at >= start_date
        ).group_by(func.date(RoutingRequest.created_at)).all()
        
        # Priority preferences
        priority_stats = self.db.query(
            RoutingRequest.user_priority,
            func.count(RoutingRequest.id).label('count')
        ).filter(
            RoutingRequest.created_at >= start_date
        ).group_by(RoutingRequest.user_priority).all()
        
        priority_total = sum(stat.count for stat in priority_stats)
        priority_distribution = {
            stat.user_priority: round((stat.count / priority_total) * 100, 1)
            for stat in priority_stats
        } if priority_total > 0 else {}
        
        # Calculate savings (comparing to always using GPT-5)
        gpt5_cost_per_token = 0.015 / 1000  # GPT-5 cost per token
        total_tokens = self.db.query(func.sum(RoutingRequest.tokens_total)).filter(
            RoutingRequest.created_at >= start_date
        ).scalar() or 0
        
        theoretical_gpt5_cost = total_tokens * gpt5_cost_per_token
        savings_percentage = ((theoretical_gpt5_cost - total_cost) / theoretical_gpt5_cost * 100) if theoretical_gpt5_cost > 0 else 0
        
        return {
            "total_requests": total_requests,
            "total_cost": total_cost,
            "avg_latency": round(avg_latency),
            "cost_savings": round(savings_percentage, 1),
            "variant_distribution": variant_distribution,
            "daily_stats": [
                {
                    "date": stat.date.isoformat(),
                    "requests": stat.requests,
                    "cost": round(stat.cost, 2),
                    "avg_latency": round(stat.avg_latency),
                    "savings": round(savings_percentage, 1)  # Simplified
                }
                for stat in daily_stats
            ],
            "priority_preferences": priority_distribution
        }
    
    def get_model_performance(self, variant: str, days: int = 30) -> Dict[str, Any]:
        """Get performance metrics for a specific model variant"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        stats = self.db.query(
            func.count(RoutingRequest.id).label('request_count'),
            func.avg(RoutingRequest.actual_cost).label('avg_cost'),
            func.avg(RoutingRequest.actual_latency_ms).label('avg_latency'),
            func.avg(RoutingRequest.quality_estimate).label('avg_quality'),
            func.sum(RoutingRequest.success.cast(Integer)).label('success_count')
        ).filter(
            and_(
                RoutingRequest.selected_variant == variant,
                RoutingRequest.created_at >= start_date
            )
        ).first()
        
        success_rate = (stats.success_count / stats.request_count * 100) if stats.request_count > 0 else 0
        
        return {
            "variant": variant,
            "request_count": stats.request_count or 0,
            "avg_cost": round(stats.avg_cost or 0, 4),
            "avg_latency": round(stats.avg_latency or 0),
            "avg_quality": round(stats.avg_quality or 0, 1),
            "success_rate": round(success_rate, 1)
        }
    
    def create_or_update_session(self, session_id: str, user_preferences: Dict[str, Any]) -> UserSession:
        """Create or update a user session"""
        session = self.db.query(UserSession).filter(
            UserSession.session_id == session_id
        ).first()
        
        if session:
            # Update existing session
            session.last_active_at = datetime.utcnow()
            session.default_priority = user_preferences.get('priority', session.default_priority)
            session.default_max_cost = user_preferences.get('max_cost', session.default_max_cost)
            session.default_max_latency = user_preferences.get('max_latency', session.default_max_latency)
        else:
            # Create new session
            session = UserSession(
                session_id=session_id,
                user_id=user_preferences.get('user_id', 'anonymous'),
                default_priority=user_preferences.get('priority', 'quality'),
                default_max_cost=user_preferences.get('max_cost', 0.02),
                default_max_latency=user_preferences.get('max_latency', 5000),
                expires_at=datetime.utcnow() + timedelta(days=30)
            )
            self.db.add(session)
        
        self.db.commit()
        self.db.refresh(session)
        return session
    
    def get_active_models(self) -> List[LLMModel]:
        """Get all active LLM models"""
        return self.db.query(LLMModel).filter(LLMModel.is_active == True).all()
    
    def record_system_metric(self, name: str, value: float, metric_type: str = "gauge", tags: Dict[str, Any] = None):
        """Record a system metric"""
        metric = SystemMetrics(
            metric_name=name,
            metric_value=value,
            metric_type=metric_type,
            tags=tags or {}
        )
        
        self.db.add(metric)
self.db.commit()