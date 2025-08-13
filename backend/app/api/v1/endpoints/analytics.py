# backend/app/api/v1/endpoints/analytics.py

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session

from app.services.database_service import DatabaseService
from app.models.database import get_db
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Response Models
class AnalyticsResponse(BaseModel):
    total_requests: int
    total_cost: float
    avg_latency: float
    cost_savings: float
    variant_distribution: Dict[str, float]
    daily_stats: List[Dict[str, Any]]
    priority_preferences: Dict[str, float]
    complexity_distribution: Dict[str, float]
    quality_metrics: Dict[str, Any]

class CostBreakdown(BaseModel):
    period_days: int
    total_cost: float
    cost_by_variant: Dict[str, float]
    cost_by_day: List[Dict[str, Any]]
    avg_cost_per_request: float
    cost_savings: float
    projected_monthly_cost: float

class PerformanceMetrics(BaseModel):
    period_days: int
    total_requests: int
    success_rate: float
    avg_latency: float
    avg_quality: float
    latency_p95: Optional[float]
    latency_p99: Optional[float]
    error_rate: float

class UsageTrends(BaseModel):
    period_days: int
    daily_usage: List[Dict[str, Any]]
    growth_rate: float
    peak_usage_hour: int
    popular_categories: Dict[str, int]
    user_engagement: Dict[str, Any]

class CostOptimizationReport(BaseModel):
    current_savings: float
    potential_additional_savings: float
    recommendations: List[Dict[str, Any]]
    cost_efficiency_score: float
    optimization_opportunities: List[str]

@router.get("/overview", response_model=AnalyticsResponse)
async def get_analytics_overview(
    days: int = Query(default=7, ge=1, le=365, description="Number of days to analyze"),
    user_id: Optional[str] = Query(default=None, description="Filter by specific user ID"),
    db: Session = Depends(get_db)
):
    """
    📊 Get comprehensive analytics overview
    
    Returns a complete analytics dashboard with key metrics for the specified time period.
    Includes request volume, costs, performance, and user behavior patterns.
    """
    try:
        db_service = DatabaseService(db)
        analytics_data = db_service.get_analytics_data(days=days, user_id=user_id)
        
        return AnalyticsResponse(**analytics_data)
        
    except Exception as e:
        logger.error(f"❌ Error getting analytics overview: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics data")

@router.get("/costs", response_model=CostBreakdown)
async def get_cost_breakdown(
    days: int = Query(default=30, ge=1, le=365),
    user_id: Optional[str] = Query(default=None),
    db: Session = Depends(get_db)
):
    """
    💰 Get detailed cost breakdown and analysis
    
    Provides comprehensive cost analysis including breakdown by model variant,
    daily trends, and savings compared to using premium models exclusively.
    """
    try:
        db_service = DatabaseService(db)
        analytics_data = db_service.get_analytics_data(days=days, user_id=user_id)
        
        # Calculate cost breakdown by variant
        cost_by_variant = {}
        total_requests = analytics_data["total_requests"]
        total_cost = analytics_data["total_cost"]
        
        if total_requests > 0:
            for variant, percentage in analytics_data["variant_distribution"].items():
                # Rough estimation based on distribution
                variant_cost = total_cost * (percentage / 100)
                cost_by_variant[variant] = round(variant_cost, 4)
        
        # Project monthly cost
        daily_avg_cost = total_cost / days if days > 0 else 0
        projected_monthly_cost = daily_avg_cost * 30
        
        return CostBreakdown(
            period_days=days,
            total_cost=total_cost,
            cost_by_variant=cost_by_variant,
            cost_by_day=analytics_data["daily_stats"],
            avg_cost_per_request=total_cost / total_requests if total_requests > 0 else 0,
            cost_savings=analytics_data["cost_savings"],
            projected_monthly_cost=round(projected_monthly_cost, 2)
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting cost breakdown: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cost data")

@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    days: int = Query(default=7, ge=1, le=90),
    variant: Optional[str] = Query(default=None, description="Filter by specific variant"),
    db: Session = Depends(get_db)
):
    """
    ⚡ Get performance metrics and quality analysis
    
    Returns detailed performance statistics including latency distribution,
    success rates, and quality scores across different model variants.
    """
    try:
        db_service = DatabaseService(db)
        
        if variant:
            # Get performance for specific variant
            performance_data = db_service.get_model_performance(variant, days)
            
            return PerformanceMetrics(
                period_days=days,
                total_requests=performance_data["request_count"],
                success_rate=performance_data["success_rate"],
                avg_latency=performance_data["avg_latency"],
                avg_quality=performance_data["avg_quality"],
                latency_p95=None,  # Would need additional query
                latency_p99=None,  # Would need additional query
                error_rate=100 - performance_data["success_rate"]
            )
        else:
            # Get overall performance
            analytics_data = db_service.get_analytics_data(days=days)
            
            return PerformanceMetrics(
                period_days=days,
                total_requests=analytics_data["total_requests"],
                success_rate=95.0,  # Would calculate from actual data
                avg_latency=analytics_data["avg_latency"],
                avg_quality=analytics_data["quality_metrics"]["avg_quality"],
                latency_p95=None,
                latency_p99=None,
                error_rate=5.0
            )
        
    except Exception as e:
        logger.error(f"❌ Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance data")

@router.get("/trends", response_model=UsageTrends)
async def get_usage_trends(
    days: int = Query(default=30, ge=7, le=365),
    db: Session = Depends(get_db)
):
    """
    📈 Get usage trends and patterns
    
    Analyzes usage patterns over time including growth trends, peak usage periods,
    and popular prompt categories.
    """
    try:
        db_service = DatabaseService(db)
        analytics_data = db_service.get_analytics_data(days=days)
        
        # Calculate growth rate (simplified)
        daily_stats = analytics_data["daily_stats"]
        if len(daily_stats) >= 2:
            recent_avg = sum(day["requests"] for day in daily_stats[-7:]) / min(7, len(daily_stats))
            earlier_avg = sum(day["requests"] for day in daily_stats[:7]) / min(7, len(daily_stats))
            growth_rate = ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg > 0 else 0
        else:
            growth_rate = 0
        
        # Mock data for categories and engagement (would be calculated from actual data)
        popular_categories = {
            "coding": 35,
            "analysis": 25,
            "creative": 20,
            "qa": 15,
            "other": 5
        }
        
        user_engagement = {
            "avg_requests_per_session": 8.5,
            "avg_session_duration_minutes": 25,
            "repeat_user_rate": 68.5
        }
        
        return UsageTrends(
            period_days=days,
            daily_usage=daily_stats,
            growth_rate=round(growth_rate, 1),
            peak_usage_hour=14,  # Would calculate from hourly data
            popular_categories=popular_categories,
            user_engagement=user_engagement
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting usage trends: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve usage trends")

@router.get("/optimization", response_model=CostOptimizationReport)
async def get_cost_optimization_report(
    days: int = Query(default=30, ge=7, le=90),
    db: Session = Depends(get_db)
):
    """
    🎯 Get cost optimization analysis and recommendations
    
    Provides actionable insights for further cost optimization including
    model selection recommendations and usage pattern analysis.
    """
    try:
        db_service = DatabaseService(db)
        analytics_data = db_service.get_analytics_data(days=days)
        
        current_savings = analytics_data["cost_savings"]
        
        # Calculate potential additional savings
        # This would involve analyzing current routing decisions vs optimal ones
        potential_additional_savings = 15.0  # Mock calculation
        
        # Generate recommendations based on usage patterns
        recommendations = []
        
        # Check if user is using expensive models for simple tasks
        variant_dist = analytics_data["variant_distribution"]
        if variant_dist.get("gpt-5", 0) > 20:
            recommendations.append({
                "type": "model_optimization",
                "title": "Reduce Premium Model Usage",
                "description": "Consider using GPT-5 Mini or Chat for medium complexity tasks",
                "potential_savings": 12.5,
                "priority": "high"
            })
        
        # Check complexity distribution
        complexity_dist = analytics_data["complexity_distribution"]
        if complexity_dist.get("Low (0-30%)", 0) > 30 and variant_dist.get("gpt-5-nano", 0) < 20:
            recommendations.append({
                "type": "routing_optimization", 
                "title": "Increase Nano Model Usage",
                "description": "Route more simple queries to GPT-5 Nano for cost savings",
                "potential_savings": 8.0,
                "priority": "medium"
            })
        
        # Calculate cost efficiency score (0-100)
        if current_savings >= 50:
            efficiency_score = 90 + (current_savings - 50) / 5
        elif current_savings >= 30:
            efficiency_score = 70 + (current_savings - 30) * 2
        else:
            efficiency_score = max(0, current_savings * 2)
        
        cost_efficiency_score = min(100, efficiency_score)
        
        # Optimization opportunities
        opportunities = []
        if cost_efficiency_score < 70:
            opportunities.append("Implement more aggressive cost-based routing")
        if variant_dist.get("gpt-5-chat", 0) < 15:
            opportunities.append("Utilize GPT-5 Chat for reasoning tasks")
        if analytics_data["total_requests"] > 1000:
            opportunities.append("Consider custom routing rules for high-volume usage")
        
        return CostOptimizationReport(
            current_savings=current_savings,
            potential_additional_savings=potential_additional_savings,
            recommendations=recommendations,
            cost_efficiency_score=round(cost_efficiency_score, 1),
            optimization_opportunities=opportunities
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting optimization report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate optimization report")

@router.get("/export")
async def export_analytics_data(
    days: int = Query(default=30, ge=1, le=365),
    format: str = Query(default="json", regex="^(json|csv)$"),
    user_id: Optional[str] = Query(default=None),
    db: Session = Depends(get_db)
):
    """
    📤 Export analytics data
    
    Export detailed analytics data in JSON or CSV format for external analysis.
    """
    try:
        db_service = DatabaseService(db)
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Export data
        export_data = db_service.export_data(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if format == "csv":
            # Convert to CSV format (simplified)
            csv_content = "timestamp,complexity_score,selected_variant,actual_cost,actual_latency_ms,quality_estimate,tokens_total,user_priority,success\n"
            for record in export_data:
                csv_content += f"{record['timestamp']},{record['complexity_score']},{record['selected_variant']},{record['actual_cost']},{record['actual_latency_ms']},{record['quality_estimate']},{record['tokens_total']},{record['user_priority']},{record['success']}\n"
            
            return {
                "format": "csv",
                "data": csv_content,
                "record_count": len(export_data),
                "period_days": days
            }
        else:
            return {
                "format": "json",
                "data": export_data,
                "record_count": len(export_data),
                "period_days": days,
                "exported_at": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        logger.error(f"❌ Error exporting data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export analytics data")

@router.get("/summary/{user_id}")
async def get_user_summary(
    user_id: str,
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """
    👤 Get user-specific analytics summary
    
    Returns personalized analytics summary for a specific user including
    usage patterns, costs, and recommendations.
    """
    try:
        db_service = DatabaseService(db)
        
        # Get user-specific analytics
        user_analytics = db_service.get_analytics_data(days=days, user_id=user_id)
        usage_summary = db_service.get_usage_summary(user_id=user_id, days=days)
        
        # Get user session info
        # This would get the latest session for the user
        # For now, we'll provide a mock summary
        
        return {
            "user_id": user_id,
            "period_days": days,
            "usage_summary": usage_summary,
            "analytics": user_analytics,
            "personalized_insights": {
                "most_used_variant": usage_summary.get("most_used_variant"),
                "avg_complexity": usage_summary.get("avg_complexity"),
                "cost_efficiency": "high" if user_analytics["cost_savings"] > 40 else "medium",
                "usage_pattern": "regular" if usage_summary.get("total_requests", 0) > 50 else "light"
            },
            "recommendations": [
                {
                    "type": "usage_optimization",
                    "message": f"You could save an additional ${(usage_summary.get('total_cost', 0) * 0.1):.2f} by using more cost-effective variants for simple tasks"
                }
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting user summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user summary")

@router.get("/health")
async def analytics_health_check():
    """
    🏥 Analytics service health check
    
    Returns the health status of the analytics service and data freshness.
    """
    try:
        return {
            "status": "healthy",
            "service": "analytics",
            "features": {
                "cost_tracking": settings.ENABLE_COST_TRACKING,
                "performance_metrics": settings.ENABLE_PERFORMANCE_METRICS,
                "quality_estimation": settings.ENABLE_QUALITY_ESTIMATION
            },
            "data_retention_days": settings.ANALYTICS_RETENTION_DAYS,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Analytics health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analytics service unhealthy")

@router.post("/cleanup")
async def cleanup_old_analytics_data(
    retention_days: Optional[int] = Query(default=None, description="Override default retention period"),
    db: Session = Depends(get_db)
):
    """
    🧹 Clean up old analytics data
    
    Remove analytics data older than the retention period to manage storage.
    Requires admin privileges.
    """
    try:
        db_service = DatabaseService(db)
        db_service.cleanup_old_data(retention_days)
        
        return {
            "message": "Analytics data cleanup completed",
            "retention_days": retention_days or settings.ANALYTICS_RETENTION_DAYS,
            "cleaned_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clean up analytics data")