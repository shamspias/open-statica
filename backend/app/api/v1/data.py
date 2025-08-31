from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Request
from typing import Optional, List
import pandas as pd
import numpy as np
import io
from app.core.session_manager import SessionManager
from app.core.base import DataInfo
from app.models import DataUploadResponse

router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


@router.post("/upload", response_model=DataUploadResponse)
async def upload_data(
        file: UploadFile = File(...),
        session_manager: SessionManager = Depends(get_session_manager)
):
    """Upload and process data file"""
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls', '.json', '.parquet')):
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Read file content
        contents = await file.read()

        # Parse based on file type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.BytesIO(contents))
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(contents))

        # Create session
        session_id = session_manager.create_session()
        session = session_manager.get_session(session_id)

        # Analyze data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        # Calculate missing values
        missing_values = df.isnull().sum().to_dict()

        # Create data info
        data_info = DataInfo(
            rows=len(df),
            columns=len(df.columns),
            column_names=df.columns.tolist(),
            column_types={col: str(df[col].dtype) for col in df.columns},
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            missing_values=missing_values,
            memory_usage=df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        )

        # Store in session
        session.set_data(df, data_info.dict())

        return DataUploadResponse(
            session_id=session_id,
            rows=data_info.rows,
            columns=data_info.columns,
            column_names=data_info.column_names,
            numeric_columns=data_info.numeric_columns,
            categorical_columns=data_info.categorical_columns,
            preview=df.head(10).replace({np.nan: None}).to_dict(orient='records')
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}")
async def get_data(
        session_id: str,
        rows: Optional[int] = None,
        columns: Optional[List[str]] = None,
        session_manager: SessionManager = Depends(get_session_manager)
):
    """Get session data"""
    session = session_manager.get_session(session_id)
    if not session or session.data is None:
        raise HTTPException(status_code=404, detail="Session not found")

    df = session.data

    # Filter columns if specified
    if columns:
        df = df[columns]

    # Limit rows if specified
    if rows:
        df = df.head(rows)

    return {
        "data": df.replace({np.nan: None}).to_dict(orient='records'),
        "shape": df.shape,
        "columns": df.columns.tolist()
    }


@router.get("/{session_id}/info")
async def get_data_info(
        session_id: str,
        session_manager: SessionManager = Depends(get_session_manager)
):
    """Get detailed data information"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session.metadata


@router.post("/{session_id}/transform")
async def transform_data(
        session_id: str,
        transformation: dict,
        session_manager: SessionManager = Depends(get_session_manager)
):
    """Apply transformations to data"""
    session = session_manager.get_session(session_id)
    if not session or session.data is None:
        raise HTTPException(status_code=404, detail="Session not found")

    df = session.data.copy()
    transform_type = transformation.get("type")

    try:
        if transform_type == "normalize":
            from sklearn.preprocessing import StandardScaler
            columns = transformation.get("columns", df.select_dtypes(include=[np.number]).columns)
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])

        elif transform_type == "encode":
            columns = transformation.get("columns", df.select_dtypes(include=['object']).columns)
            df = pd.get_dummies(df, columns=columns)

        elif transform_type == "impute":
            strategy = transformation.get("strategy", "mean")
            columns = transformation.get("columns", df.columns)

            if strategy == "mean":
                df[columns] = df[columns].fillna(df[columns].mean())
            elif strategy == "median":
                df[columns] = df[columns].fillna(df[columns].median())
            elif strategy == "mode":
                df[columns] = df[columns].fillna(df[columns].mode().iloc[0])
            elif strategy == "forward":
                df[columns] = df[columns].fillna(method='ffill')
            elif strategy == "backward":
                df[columns] = df[columns].fillna(method='bfill')

        # Update session data
        session.data = df

        return {"message": "Transformation applied successfully", "new_shape": df.shape}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def delete_session(
        session_id: str,
        session_manager: SessionManager = Depends(get_session_manager)
):
    """Delete a session and its data"""
    if session_manager.delete_session(session_id):
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")
