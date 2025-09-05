from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
from typing import Dict, Any
import uuid
from datetime import datetime

from app.config import get_settings
from app.services.statistics_service import StatisticsService
from app.services.visualization_service import VisualizationService

settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Simple statistical analysis platform for researchers",
    version=settings.APP_VERSION
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
stats_service = StatisticsService()
viz_service = VisualizationService()

# In-memory session storage (for simplicity)
sessions = {}


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload and process data file"""
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")

        # Read file
        contents = await file.read()

        # Parse based on file type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Store in session
        sessions[session_id] = {
            'data': df,
            'created_at': datetime.now(),
            'filename': file.filename
        }

        # Get basic info
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        return {
            'session_id': session_id,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'preview': df.head(10).replace({np.nan: None}).to_dict(orient='records')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/{session_id}")
async def get_data(session_id: str):
    """Get session data"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[session_id]['data']
    return {
        'data': df.replace({np.nan: None}).to_dict(orient='records'),
        'columns': df.columns.tolist(),
        'shape': df.shape
    }


@app.post("/api/statistics/descriptive")
async def descriptive_statistics(request: Dict[str, Any]):
    """Calculate descriptive statistics"""
    session_id = request.get('session_id')
    columns = request.get('columns', [])

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[session_id]['data']
    result = await stats_service.descriptive_statistics(df, columns)

    return result


@app.post("/api/statistics/correlation")
async def correlation_analysis(request: Dict[str, Any]):
    """Calculate correlations"""
    session_id = request.get('session_id')
    columns = request.get('columns', [])
    method = request.get('method', 'pearson')

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[session_id]['data']
    result = await stats_service.correlation_analysis(df, columns, method)

    return result


@app.post("/api/statistics/ttest")
async def t_test(request: Dict[str, Any]):
    """Perform t-test"""
    session_id = request.get('session_id')

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[session_id]['data']
    result = await stats_service.t_test(df, request)

    return result


@app.post("/api/statistics/anova")
async def anova(request: Dict[str, Any]):
    """Perform ANOVA"""
    session_id = request.get('session_id')

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[session_id]['data']
    result = await stats_service.anova(df, request)

    return result


@app.post("/api/statistics/regression")
async def regression(request: Dict[str, Any]):
    """Perform regression analysis"""
    session_id = request.get('session_id')

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[session_id]['data']
    result = await stats_service.regression(df, request)

    return result


@app.post("/api/statistics/chi-square")
async def chi_square(request: Dict[str, Any]):
    """Perform chi-square test"""
    session_id = request.get('session_id')

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[session_id]['data']
    result = await stats_service.chi_square_test(df, request)

    return result


@app.post("/api/visualization/create")
async def create_visualization(request: Dict[str, Any]):
    """Create visualization"""
    session_id = request.get('session_id')
    chart_type = request.get('type')

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[session_id]['data']
    result = await viz_service.create_chart(df, chart_type, request)

    return result


@app.post("/api/ml/cluster")
async def clustering(request: Dict[str, Any]):
    """Perform clustering analysis"""
    session_id = request.get('session_id')

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[session_id]['data']
    result = await stats_service.clustering(df, request)

    return result


@app.post("/api/ml/classify")
async def classification(request: Dict[str, Any]):
    """Train classification model"""
    session_id = request.get('session_id')

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[session_id]['data']
    result = await stats_service.classification(df, request)

    return result


@app.post("/api/export")
async def export_data(request: Dict[str, Any]):
    """Export results"""
    session_id = request.get('session_id')
    format = request.get('format', 'csv')

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[session_id]['data']

    if format == 'csv':
        output = df.to_csv(index=False)
        return {'data': output, 'format': 'csv'}
    elif format == 'excel':
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        return {'data': output.read(), 'format': 'excel'}
    else:
        return {'data': df.to_json(orient='records'), 'format': 'json'}
