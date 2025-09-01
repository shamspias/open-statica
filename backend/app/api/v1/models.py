from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Dict, Any, List, Optional
from app.core.session_manager import SessionManager
from app.core.registry import EngineRegistry
from app.engines.ml.model_hub import ModelHubEngine, PretrainedModelRegistry

router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def get_registry(request: Request) -> EngineRegistry:
    return request.app.state.registry


@router.get("/available")
async def list_available_models():
    """List all available pretrained models"""
    return PretrainedModelRegistry.get_all_models()


@router.get("/search")
async def search_models(
        q: str,
        source: str = "huggingface",
        task: Optional[str] = None
):
    """Search for models"""
    if task:
        models = PretrainedModelRegistry.get_models_by_task(task)
        # Filter by search query
        models = [m for m in models if q.lower() in m.lower()]
    else:
        # Return all models matching query
        models = []
        for category in PretrainedModelRegistry.MODELS.values():
            for task_models in category.values():
                models.extend([m for m in task_models if q.lower() in m.lower()])

    return {
        "query": q,
        "source": source,
        "task": task,
        "results": models
    }


@router.post("/load")
async def load_model(
        request: Dict[str, Any],
        session_manager: SessionManager = Depends(get_session_manager),
        registry: EngineRegistry = Depends(get_registry)
):
    """Load a pretrained model from model hub"""
    session_id = request.get("session_id")
    model_id = request.get("model_id")
    source = request.get("source", "huggingface")
    task = request.get("task")

    session = session_manager.get_session(session_id) if session_id else None

    try:
        # Create model hub engine
        engine = ModelHubEngine(source)

        # Register engine
        await registry.register(engine)

        # Load model
        result = await engine.load_model(model_id, task)

        # Store in session if available
        if session:
            import uuid
            internal_model_id = str(uuid.uuid4())
            session.add_model(internal_model_id, engine)
            result["internal_model_id"] = internal_model_id

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference")
async def model_inference(
        request: Dict[str, Any],
        session_manager: SessionManager = Depends(get_session_manager)
):
    """Run inference with a loaded model"""
    session_id = request.get("session_id")
    model_id = request.get("model_id")
    input_data = request.get("input")

    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if model_id not in session.models:
        raise HTTPException(status_code=404, detail="Model not found in session")

    try:
        engine = session.models[model_id]

        # Run inference
        predictions = await engine.predict(input_data)

        return {
            "model_id": model_id,
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/finetune")
async def finetune_model(
        request: Dict[str, Any],
        session_manager: SessionManager = Depends(get_session_manager)
):
    """Fine-tune a pretrained model"""
    session_id = request.get("session_id")
    model_id = request.get("model_id")
    training_data = request.get("training_data")
    training_params = request.get("params", {})

    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if model_id not in session.models:
        raise HTTPException(status_code=404, detail="Model not found in session")

    try:
        engine = session.models[model_id]

        # Fine-tune model
        result = await engine.train(
            training_data.get("X"),
            training_data.get("y"),
            **training_params
        )

        return {
            "model_id": model_id,
            "status": "fine-tuned",
            "results": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hub/{source}")
async def get_hub_info(source: str):
    """Get information about a specific model hub"""
    hub_info = {
        "huggingface": {
            "name": "Hugging Face Model Hub",
            "url": "https://huggingface.co/models",
            "description": "The largest hub of ready-to-use models for NLP, computer vision, audio, and multimodal tasks",
            "models_count": "350,000+",
            "supported_tasks": [
                "text-classification", "token-classification", "question-answering",
                "summarization", "translation", "text-generation", "fill-mask",
                "image-classification", "object-detection", "image-segmentation",
                "audio-classification", "automatic-speech-recognition", "text-to-speech"
            ]
        },
        "tensorflow_hub": {
            "name": "TensorFlow Hub",
            "url": "https://tfhub.dev",
            "description": "Repository of trained machine learning models ready for fine-tuning and deployable anywhere",
            "models_count": "3,000+",
            "supported_tasks": [
                "image-classification", "object-detection", "image-segmentation",
                "text-embedding", "text-classification", "video-classification"
            ]
        },
        "torch_hub": {
            "name": "PyTorch Hub",
            "url": "https://pytorch.org/hub",
            "description": "Pre-trained models designed to facilitate research reproducibility",
            "models_count": "500+",
            "supported_tasks": [
                "image-classification", "object-detection", "semantic-segmentation",
                "image-generation", "nlp", "audio", "generative"
            ]
        }
    }

    if source not in hub_info:
        raise HTTPException(status_code=404, detail=f"Unknown model hub: {source}")

    return hub_info[source]
