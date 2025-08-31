from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any, List
from app.core.session_manager import SessionManager
from app.core.registry import EngineRegistry
from app.core.base import AnalysisRequest, AnalysisResult, EngineType
from app.engines.statistical.descriptive import DescriptiveAnalyzer, FrequencyAnalyzer
from app.engines.statistical.inferential import TTestAnalyzer, ANOVAAnalyzer

router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def get_registry(request: Request) -> EngineRegistry:
    return request.app.state.registry


@router.post("/descriptive", response_model=AnalysisResult)
async def descriptive_statistics(
        request: AnalysisRequest,
        session_manager: SessionManager = Depends(get_session_manager),
        registry: EngineRegistry = Depends(get_registry)
):
    """Calculate descriptive statistics"""
    session = session_manager.get_session(request.session_id)
    if not session or session.data is None:
        raise HTTPException(status_code=404, detail="Session or data not found")

    analyzer = registry.get("descriptive_statistics")
    if not analyzer:
        analyzer = DescriptiveAnalyzer()
        await registry.register(analyzer)

    result = await analyzer.execute(session.data, request.options)

    if result.success:
        session.add_result("descriptive", result.data)
        return AnalysisResult(
            test_name="Descriptive Statistics",
            results=result.data,
            execution_time=0.0
        )
    else:
        raise HTTPException(status_code=400, detail=result.error)


@router.post("/frequency", response_model=AnalysisResult)
async def frequency_distribution(
        request: AnalysisRequest,
        session_manager: SessionManager = Depends(get_session_manager),
        registry: EngineRegistry = Depends(get_registry)
):
    """Calculate frequency distributions"""
    session = session_manager.get_session(request.session_id)
    if not session or session.data is None:
        raise HTTPException(status_code=404, detail="Session or data not found")

    analyzer = registry.get("frequency_distribution")
    if not analyzer:
        analyzer = FrequencyAnalyzer()
        await registry.register(analyzer)

    result = await analyzer.execute(session.data, request.options)

    if result.success:
        session.add_result("frequency", result.data)
        return AnalysisResult(
            test_name="Frequency Distribution",
            results=result.data,
            execution_time=0.0
        )
    else:
        raise HTTPException(status_code=400, detail=result.error)


@router.post("/ttest", response_model=AnalysisResult)
async def t_test(
        request: AnalysisRequest,
        session_manager: SessionManager = Depends(get_session_manager),
        registry: EngineRegistry = Depends(get_registry)
):
    """Perform T-test"""
    session = session_manager.get_session(request.session_id)
    if not session or session.data is None:
        raise HTTPException(status_code=404, detail="Session or data not found")

    analyzer = registry.get("ttest")
    if not analyzer:
        analyzer = TTestAnalyzer()
        await registry.register(analyzer)

    result = await analyzer.execute(session.data, request.options)

    if result.success:
        session.add_result("ttest", result.data)
        return AnalysisResult(
            test_name="T-Test",
            results=result.data,
            execution_time=0.0
        )
    else:
        raise HTTPException(status_code=400, detail=result.error)


@router.post("/anova", response_model=AnalysisResult)
async def anova(
        request: AnalysisRequest,
        session_manager: SessionManager = Depends(get_session_manager),
        registry: EngineRegistry = Depends(get_registry)
):
    """Perform ANOVA"""
    session = session_manager.get_session(request.session_id)
    if not session or session.data is None:
        raise HTTPException(status_code=404, detail="Session or data not found")

    analyzer = registry.get("anova")
    if not analyzer:
        analyzer = ANOVAAnalyzer()
        await registry.register(analyzer)

    result = await analyzer.execute(session.data, request.options)

    if result.success:
        session.add_result("anova", result.data)
        return AnalysisResult(
            test_name="ANOVA",
            results=result.data,
            execution_time=0.0
        )
    else:
        raise HTTPException(status_code=400, detail=result.error)


@router.get("/available")
async def list_available_tests(registry: EngineRegistry = Depends(get_registry)):
    """List all available statistical tests"""
    engines = registry.get_by_type(EngineType.STATISTICAL)
    return {
        "tests": [engine.name for engine in engines],
        "categories": {
            "descriptive": ["basic_statistics", "frequency_distribution", "crosstabulation"],
            "inferential": ["ttest", "anova", "chi_square", "correlation"],
            "regression": ["linear", "logistic", "polynomial", "ridge", "lasso"],
            "nonparametric": ["mann_whitney", "wilcoxon", "kruskal_wallis", "friedman"],
            "multivariate": ["pca", "factor_analysis", "cluster", "discriminant"],
            "timeseries": ["arima", "seasonal_decomposition", "forecasting"]
        }
    }
