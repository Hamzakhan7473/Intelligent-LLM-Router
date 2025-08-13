# backend/app/api/v1/endpoints/evaluation.py

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import uuid
import logging
from sqlalchemy.orm import Session

from app.services.llm_service import LLMAPIService
from app.services.database_service import DatabaseService
from app.models.database import get_db
from app.ml.routing.gpt5_router import GPTVariant
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response Models
class EvaluationPrompt(BaseModel):
    prompt: str = Field(..., description="Test prompt")
    expected_category: Optional[str] = Field(None, description="Expected prompt category")
    expected_complexity: Optional[float] = Field(None, description="Expected complexity score (0-1)")
    gold_standard_model: Optional[str] = Field(None, description="Gold standard model for comparison")
    evaluation_criteria: List[str] = Field(default=[], description="Criteria to evaluate")

class EvaluationRequest(BaseModel):
    prompts: List[EvaluationPrompt] = Field(..., description="List of prompts to evaluate")
    variants_to_test: Optional[List[str]] = Field(None, description="Specific variants to test")
    include_quality_analysis: bool = Field(default=True, description="Include quality analysis")
    include_cost_analysis: bool = Field(default=True, description="Include cost analysis")
    test_name: Optional[str] = Field(None, description="Name for this evaluation")

class EvaluationResult(BaseModel):
    prompt: str
    expected_category: Optional[str]
    actual_category: Optional[str]
    expected_complexity: Optional[float]
    actual_complexity: float
    routing_decision: Dict[str, Any]
    responses: Dict[str, Dict[str, Any]]  # variant -> response data
    quality_scores: Dict[str, float]
    cost_comparison: Dict[str, float]
    recommendation: Dict[str, Any]

class EvaluationReport(BaseModel):
    evaluation_id: str
    test_name: Optional[str]
    total_prompts: int
    variants_tested: List[str]
    results: List[EvaluationResult]
    summary: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    created_at: datetime
    status: str

class BenchmarkRequest(BaseModel):
    benchmark_name: str = Field(..., description="Name of the benchmark")
    variants: List[str] = Field(..., description="Variants to benchmark")
    metrics: List[str] = Field(default=["accuracy", "cost", "latency"], description="Metrics to measure")

class RoutingAccuracyTest(BaseModel):
    test_cases: List[Dict[str, Any]] = Field(..., description="Test cases with expected routing decisions")
    tolerance: float = Field(default=0.1, description="Acceptable difference in complexity scores")

# Predefined benchmark datasets
BENCHMARK_DATASETS = {
    "coding_tasks": [
        {
            "prompt": "Write a Python function to implement quicksort",
            "expected_category": "coding",
            "expected_complexity": 0.5,
            "gold_standard_model": "gpt-5"
        },
        {
            "prompt": "Create a simple hello world program",
            "expected_category": "coding", 
            "expected_complexity": 0.2,
            "gold_standard_model": "gpt-5-mini"
        },
        {
            "prompt": "Build a complete web application with authentication and database",
            "expected_category": "coding",
            "expected_complexity": 0.9,
            "gold_standard_model": "gpt-5"
        }
    ],
    "reasoning_tasks": [
        {
            "prompt": "Think step by step: Why did the Roman Empire fall?",
            "expected_category": "reasoning",
            "expected_complexity": 0.8,
            "gold_standard_model": "gpt-5-chat"
        },
        {
            "prompt": "What is 2+2?",
            "expected_category": "simple_qa",
            "expected_complexity": 0.1,
            "gold_standard_model": "gpt-5-nano"
        },
        {
            "prompt": "Analyze the economic implications of climate change on global trade",
            "expected_category": "analysis",
            "expected_complexity": 0.85,
            "gold_standard_model": "gpt-5"
        }
    ],
    "creative_tasks": [
        {
            "prompt": "Write a short story about a robot learning to love",
            "expected_category": "creative",
            "expected_complexity": 0.6,
            "gold_standard_model": "gpt-5"
        },
        {
            "prompt": "Come up with a name for my cat",
            "expected_category": "creative",
            "expected_complexity": 0.2,
            "gold_standard_model": "gpt-5-mini"
        }
    ]
}

@router.post("/run", response_model=EvaluationReport)
async def run_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    🧪 Run comprehensive model evaluation
    
    Evaluates multiple model variants against a set of test prompts,
    measuring accuracy, cost, latency, and quality metrics.
    """
    evaluation_id = str(uuid.uuid4())
    
    try:
        # Default to all GPT-5 variants if none specified
        variants_to_test = request.variants_to_test or [variant.value for variant in GPTVariant]
        
        # Validate variants
        valid_variants = [variant.value for variant in GPTVariant]
        invalid_variants = [v for v in variants_to_test if v not in valid_variants]
        if invalid_variants:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid variants: {invalid_variants}"
            )
        
        logger.info(f"🧪 Starting evaluation {evaluation_id} with {len(request.prompts)} prompts")
        
        results = []
        
        async with LLMAPIService() as llm_service:
            # Process each prompt
            for prompt_data in request.prompts:
                logger.info(f"🔄 Evaluating prompt: {prompt_data.prompt[:50]}...")
                
                # Get routing decision for this prompt
                routing_result = await llm_service.router.route_with_analysis(
                    prompt_data.prompt,
                    {"priority": "quality"}  # Use quality priority for evaluation
                )
                
                # Test each variant
                variant_responses = {}
                quality_scores = {}
                cost_comparison = {}
                
                for variant_str in variants_to_test:
                    try:
                        variant = GPTVariant(variant_str)
                        
                        # Make API call
                        response = await llm_service.call_llm_with_fallback(
                            variant=variant,
                            prompt=prompt_data.prompt
                        )
                        
                        variant_responses[variant_str] = {
                            "content": response.content[:200],  # Truncate for storage
                            "cost": response.cost,
                            "latency_ms": response.latency_ms,
                            "tokens_used": response.tokens_used,
                            "quality_estimate": response.quality_estimate,
                            "success": response.success
                        }
                        
                        quality_scores[variant_str] = response.quality_estimate
                        cost_comparison[variant_str] = response.cost
                        
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to test {variant_str}: {str(e)}")
                        variant_responses[variant_str] = {"error": str(e), "success": False}
                        quality_scores[variant_str] = 0.0
                        cost_comparison[variant_str] = 0.0
                
                # Generate recommendation for this prompt
                successful_variants = {k: v for k, v in variant_responses.items() if v.get("success", False)}
                
                if successful_variants:
                    # Find best variant based on different criteria
                    best_quality = max(successful_variants.items(), key=lambda x: x[1]["quality_estimate"])
                    best_cost = min(successful_variants.items(), key=lambda x: x[1]["cost"])
                    best_speed = min(successful_variants.items(), key=lambda x: x[1]["latency_ms"])
                    
                    recommendation = {
                        "best_quality": best_quality[0],
                        "best_cost": best_cost[0],
                        "best_speed": best_speed[0],
                        "routing_decision": routing_result.selected_variant.value,
                        "routing_accuracy": routing_result.selected_variant.value == prompt_data.gold_standard_model
                    }
                else:
                    recommendation = {"error": "All variants failed"}
                
                # Create result
                result = EvaluationResult(
                    prompt=prompt_data.prompt,
                    expected_category=prompt_data.expected_category,
                    actual_category=routing_result.routing_metadata.get("domain"),
                    expected_complexity=prompt_data.expected_complexity,
                    actual_complexity=routing_result.complexity_score,
                    routing_decision={
                        "selected_variant": routing_result.selected_variant.value,
                        "confidence": routing_result.confidence,
                        "reasoning": routing_result.reasoning
                    },
                    responses=variant_responses,
                    quality_scores=quality_scores,
                    cost_comparison=cost_comparison,
                    recommendation=recommendation
                )
                
                results.append(result)
        
        # Generate summary
        summary = generate_evaluation_summary(results, variants_to_test)
        recommendations = generate_evaluation_recommendations(results)
        
        # Create evaluation report
        report = EvaluationReport(
            evaluation_id=evaluation_id,
            test_name=request.test_name,
            total_prompts=len(request.prompts),
            variants_tested=variants_to_test,
            results=results,
            summary=summary,
            recommendations=recommendations,
            created_at=datetime.utcnow(),
            status="completed"
        )
        
        # Store evaluation results in background
        background_tasks.add_task(
            store_evaluation_results,
            evaluation_id,
            report,
            db
        )
        
        logger.info(f"✅ Evaluation {evaluation_id} completed successfully")
        return report
        
    except Exception as e:
        logger.error(f"❌ Evaluation {evaluation_id} failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}" if settings.DEBUG else "Evaluation failed"
        )

@router.post("/benchmark/{benchmark_name}")
async def run_benchmark(
    benchmark_name: str,
    variants: List[str],
    db: Session = Depends(get_db)
):
    """
    📊 Run predefined benchmark test
    
    Executes a standardized benchmark test with predefined prompts and evaluation criteria.
    """
    try:
        if benchmark_name not in BENCHMARK_DATASETS:
            raise HTTPException(
                status_code=404,
                detail=f"Benchmark '{benchmark_name}' not found. Available: {list(BENCHMARK_DATASETS.keys())}"
            )
        
        # Get benchmark prompts
        benchmark_prompts = BENCHMARK_DATASETS[benchmark_name]
        
        # Convert to evaluation prompts
        eval_prompts = [
            EvaluationPrompt(**prompt_data) for prompt_data in benchmark_prompts
        ]
        
        # Create evaluation request
        eval_request = EvaluationRequest(
            prompts=eval_prompts,
            variants_to_test=variants,
            test_name=f"Benchmark: {benchmark_name}"
        )
        
        # Run evaluation
        report = await run_evaluation(eval_request, BackgroundTasks(), db)
        
        # Add benchmark-specific analysis
        benchmark_analysis = analyze_benchmark_results(report, benchmark_name)
        
        return {
            "benchmark_name": benchmark_name,
            "evaluation_report": report,
            "benchmark_analysis": benchmark_analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Benchmark {benchmark_name} failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Benchmark execution failed")

@router.post("/routing-accuracy")
async def test_routing_accuracy(
    request: RoutingAccuracyTest,
    db: Session = Depends(get_db)
):
    """
    🎯 Test routing accuracy against expected decisions
    
    Evaluates how well the routing algorithm matches expected routing decisions
    for a set of test cases.
    """
    try:
        results = []
        correct_decisions = 0
        total_tests = len(request.test_cases)
        
        async with LLMAPIService() as llm_service:
            for i, test_case in enumerate(request.test_cases):
                prompt = test_case["prompt"]
                expected_variant = test_case["expected_variant"]
                expected_complexity = test_case.get("expected_complexity")
                user_preferences = test_case.get("user_preferences", {"priority": "quality"})
                
                # Get routing decision
                routing_result = await llm_service.router.route_with_analysis(
                    prompt, user_preferences
                )
                
                # Check accuracy
                variant_match = routing_result.selected_variant.value == expected_variant
                
                complexity_match = True
                if expected_complexity is not None:
                    complexity_diff = abs(routing_result.complexity_score - expected_complexity)
                    complexity_match = complexity_diff <= request.tolerance
                
                is_correct = variant_match and complexity_match
                if is_correct:
                    correct_decisions += 1
                
                results.append({
                    "test_case_id": i + 1,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "expected_variant": expected_variant,
                    "actual_variant": routing_result.selected_variant.value,
                    "expected_complexity": expected_complexity,
                    "actual_complexity": routing_result.complexity_score,
                    "variant_match": variant_match,
                    "complexity_match": complexity_match,
                    "is_correct": is_correct,
                    "confidence": routing_result.confidence,
                    "reasoning": routing_result.reasoning
                })
        
        accuracy = (correct_decisions / total_tests * 100) if total_tests > 0 else 0
        
        return {
            "total_test_cases": total_tests,
            "correct_decisions": correct_decisions,
            "accuracy_percentage": round(accuracy, 1),
            "results": results,
            "summary": {
                "variant_accuracy": sum(1 for r in results if r["variant_match"]) / total_tests * 100,
                "complexity_accuracy": sum(1 for r in results if r["complexity_match"]) / total_tests * 100,
                "avg_confidence": sum(r["confidence"] for r in results) / total_tests,
                "tolerance_used": request.tolerance
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Routing accuracy test failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Routing accuracy test failed")

@router.get("/benchmarks")
async def list_available_benchmarks():
    """
    📋 List available benchmark datasets
    
    Returns information about all available benchmark datasets for evaluation.
    """
    benchmarks = {}
    for name, prompts in BENCHMARK_DATASETS.items():
        benchmarks[name] = {
            "name": name,
            "description": f"Benchmark for {name.replace('_', ' ')}",
            "prompt_count": len(prompts),
            "categories": list(set(p.get("expected_category") for p in prompts)),
            "complexity_range": {
                "min": min(p.get("expected_complexity", 0) for p in prompts),
                "max": max(p.get("expected_complexity", 1) for p in prompts)
            }
        }
    
    return {
        "available_benchmarks": benchmarks,
        "total_benchmarks": len(benchmarks)
    }

@router.get("/{evaluation_id}")
async def get_evaluation_results(evaluation_id: str, db: Session = Depends(get_db)):
    """
    📄 Get evaluation results by ID
    
    Retrieves stored evaluation results for analysis and comparison.
    """
    try:
        # This would retrieve from database in a real implementation
        # For now, return a placeholder response
        
        return {
            "evaluation_id": evaluation_id,
            "status": "completed",
            "message": "Evaluation results would be retrieved from database",
            "note": "Implementation needed: Store and retrieve evaluation results from database"
        }
        
    except Exception as e:
        logger.error(f"❌ Error retrieving evaluation {evaluation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve evaluation results")

# Helper functions
def generate_evaluation_summary(results: List[EvaluationResult], variants_tested: List[str]) -> Dict[str, Any]:
    """Generate summary statistics from evaluation results"""
    if not results:
        return {}
    
    # Calculate routing accuracy
    routing_accuracy = sum(
        1 for r in results 
        if r.recommendation.get("routing_accuracy", False)
    ) / len(results) * 100
    
    # Calculate average complexity accuracy
    complexity_accuracy = sum(
        1 for r in results 
        if r.expected_complexity and abs(r.actual_complexity - r.expected_complexity) <= 0.1
    ) / len([r for r in results if r.expected_complexity]) * 100 if any(r.expected_complexity for r in results) else 0
    
    # Calculate cost efficiency
    total_costs = {}
    for variant in variants_tested:
        variant_costs = [r.cost_comparison.get(variant, 0) for r in results]
        total_costs[variant] = sum(variant_costs)
    
    cheapest_variant = min(total_costs.items(), key=lambda x: x[1])[0] if total_costs else None
    most_expensive = max(total_costs.items(), key=lambda x: x[1])[1] if total_costs else 0
    cheapest_cost = min(total_costs.values()) if total_costs else 0
    
    cost_savings = ((most_expensive - cheapest_cost) / most_expensive * 100) if most_expensive > 0 else 0
    
    return {
        "routing_accuracy": round(routing_accuracy, 1),
        "complexity_accuracy": round(complexity_accuracy, 1),
        "cost_efficiency": {
            "cheapest_variant": cheapest_variant,
            "potential_savings": round(cost_savings, 1),
            "total_costs": total_costs
        },
        "quality_analysis": {
            "avg_quality_by_variant": {
                variant: sum(r.quality_scores.get(variant, 0) for r in results) / len(results)
                for variant in variants_tested
            }
        }
    }

def generate_evaluation_recommendations(results: List[EvaluationResult]) -> List[Dict[str, Any]]:
    """Generate actionable recommendations from evaluation results"""
    recommendations = []
    
    if not results:
        return recommendations
    
    # Analyze routing accuracy
    routing_errors = [r for r in results if not r.recommendation.get("routing_accuracy", True)]
    if len(routing_errors) > len(results) * 0.2:  # More than 20% errors
        recommendations.append({
            "type": "routing_improvement",
            "priority": "high",
            "title": "Improve Routing Accuracy",
            "description": f"Routing accuracy is {(1 - len(routing_errors)/len(results))*100:.1f}%. Consider adjusting complexity thresholds.",
            "affected_prompts": len(routing_errors)
        })
    
    # Analyze cost optimization opportunities
    cost_opportunities = []
    for result in results:
        if result.cost_comparison:
            cheapest = min(result.cost_comparison.items(), key=lambda x: x[1])
            routed = result.routing_decision["selected_variant"]
            routed_cost = result.cost_comparison.get(routed, 0)
            
            if routed_cost > cheapest[1] * 1.5:  # 50% more expensive
                cost_opportunities.append(result)
    
    if cost_opportunities:
        recommendations.append({
            "type": "cost_optimization",
            "priority": "medium",
            "title": "Cost Optimization Opportunity",
            "description": f"{len(cost_opportunities)} prompts could use cheaper variants with minimal quality impact.",
            "potential_savings": len(cost_opportunities) / len(results) * 100
        })
    
    return recommendations

def analyze_benchmark_results(report: EvaluationReport, benchmark_name: str) -> Dict[str, Any]:
    """Analyze benchmark-specific results"""
    analysis = {
        "benchmark_name": benchmark_name,
        "pass_rate": 0,
        "performance_grade": "N/A",
        "strengths": [],
        "weaknesses": []
    }
    
    if benchmark_name == "coding_tasks":
        # Analyze coding-specific metrics
        coding_results = [r for r in report.results if r.expected_category == "coding"]
        if coding_results:
            avg_quality = sum(max(r.quality_scores.values()) for r in coding_results) / len(coding_results)
            analysis["performance_grade"] = "A" if avg_quality > 8.5 else "B" if avg_quality > 7.0 else "C"
            
            if avg_quality > 8.0:
                analysis["strengths"].append("High code quality generation")
            if any(r.routing_decision["selected_variant"] == "gpt-5-mini" for r in coding_results):
                analysis["strengths"].append("Cost-effective routing for coding tasks")
    
    return analysis

async def store_evaluation_results(evaluation_id: str, report: EvaluationReport, db: Session):
    """Store evaluation results in database (background task)"""
    try:
        # This would store the evaluation results in the database
        # For now, just log the completion
        logger.info(f"✅ Stored evaluation results for {evaluation_id}")
        
        # In a real implementation, this would:
        # 1. Create evaluation record in database
        # 2. Store individual test results
        # 3. Store summary statistics
        # 4. Update model performance metrics
        
    except Exception as e:
        logger.error(f"❌ Failed to store evaluation results for {evaluation_id}: {str(e)}")