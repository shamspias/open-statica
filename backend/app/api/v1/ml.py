from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any, List
from app.core.session_manager import SessionManager
from app.core.registry import EngineRegistry
from app.engines.ml.supervised import ClassificationEngine, RegressionEngine
from app.engines.ml.unsupervised import ClusteringEngine, DimensionalityReductionEngine
from app.models import MLRequest, MLResult

router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def get_registry(request: Request) -> EngineRegistry:
    return request.app.state.registry


@router.post("/train", response_model=MLResult)
async def train_model(
        request: MLRequest,
        session_manager: SessionManager = Depends(get_session_manager),
        registry: EngineRegistry = Depends(get_registry)
):
    """Train a machine learning model"""
    session = session_manager.get_session(request.session_id)
    if not session or session.data is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        # Select appropriate engine based on task
        if request.task == "classification":
            engine = ClassificationEngine(request.algorithm)
        elif request.task == "regression":
            engine = RegressionEngine(request.algorithm)
        elif request.task == "clustering":
            engine = ClusteringEngine(request.algorithm)
        elif request.task == "dimensionality":
            engine = DimensionalityReductionEngine(request.algorithm)
        else:
            raise ValueError(f"Unknown task: {request.task}")

        # Register engine if not already registered
        if not registry.get(engine.name):
            await registry.register(engine)

        # Prepare data
        df = session.data

        if request.task in ["classification", "regression"]:
            # Supervised learning
            X = df[request.features]
            y = df[request.target]

            # Handle categorical variables
            X = pd.get_dummies(X)

            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1 - request.train_test_split, random_state=42
            )

            # Train model
            result = await engine.train(X_train, y_train, **request.options)

            # Evaluate on test set
            evaluation = await engine.evaluate(X_test, y_test)
            result.update(evaluation)

        else:
            # Unsupervised learning
            X = df[request.features]
            X = pd.get_dummies(X)

            result = await engine.execute(X, request.options)

        # Store model in session
        import uuid
        model_id = str(uuid.uuid4())
        session.add_model(model_id, engine)

        return MLResult(
            model_id=model_id,
            task=request.task,
            algorithm=request.algorithm,
            metrics=result.get("metrics", {}),
            feature_importance=result.get("feature_importance"),
            confusion_matrix=result.get("confusion_matrix"),
            execution_time=result.get("execution_time", 0)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict(
        request: Dict[str, Any],
        session_manager: SessionManager = Depends(get_session_manager)
):
    """Make predictions with trained model"""
    session = session_manager.get_session(request["session_id"])
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    model_id = request["model_id"]
    if model_id not in session.models:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        engine = session.models[model_id]
        data = pd.DataFrame(request["data"])

        predictions = await engine.predict(data)

        return {
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            "model_id": model_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def evaluate_model(
        request: Dict[str, Any],
        session_manager: SessionManager = Depends(get_session_manager)
):
    """Evaluate model performance"""
    session = session_manager.get_session(request["session_id"])
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    model_id = request["model_id"]
    if model_id not in session.models:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        engine = session.models[model_id]
        X = pd.DataFrame(request["X"])
        y = request["y"]

        evaluation = await engine.evaluate(X, y)

        return evaluation

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_available_models():
    """List all available ML models"""
    return {
        "classification": [
            "logistic", "svm", "random_forest", "xgboost",
            "neural_network", "knn", "naive_bayes", "decision_tree"
        ],
        "regression": [
            "linear", "ridge", "lasso", "elastic_net",
            "svr", "random_forest_regressor", "xgboost_regressor"
        ],
        "clustering": [
            "kmeans", "dbscan", "hierarchical",
            "gaussian_mixture", "mean_shift", "spectral"
        ],
        "dimensionality_reduction": [
            "pca", "tsne", "umap", "lda", "ica"
        ],
        "deep_learning": [
            "mlp", "cnn", "rnn", "lstm", "transformer"
        ]
    }


@router.get("/models/{model_id}")
async def get_model_info(
        model_id: str,
        session_id: str,
        session_manager: SessionManager = Depends(get_session_manager)
):
    """Get information about a specific model"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if model_id not in session.models:
        raise HTTPException(status_code=404, detail="Model not found")

    model = session.models[model_id]

    return {
        "model_id": model_id,
        "type": model.model_type,
        "trained": model.is_trained,
        "parameters": model.get_params() if hasattr(model, 'get_params') else {}
    }
