"""
Model Hub Engine for OpenStatica
Integration with Hugging Face, TensorFlow Hub, and other model repositories
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from app.core.base import BaseMLModel, Result
import os

logger = logging.getLogger(__name__)


class PretrainedModelRegistry:
    """Registry of available pretrained models"""

    MODELS = {
        "nlp": {
            "text_classification": [
                "bert-base-uncased",
                "distilbert-base-uncased",
                "roberta-base",
                "albert-base-v2",
                "xlnet-base-cased"
            ],
            "named_entity_recognition": [
                "dslim/bert-base-NER",
                "dbmdz/bert-large-cased-finetuned-conll03-english",
                "Jean-Baptiste/camembert-ner"
            ],
            "question_answering": [
                "distilbert-base-uncased-distilled-squad",
                "bert-large-uncased-whole-word-masking-squad2",
                "deepset/roberta-base-squad2"
            ],
            "summarization": [
                "facebook/bart-large-cnn",
                "google/pegasus-xsum",
                "t5-base"
            ],
            "translation": [
                "Helsinki-NLP/opus-mt-en-de",
                "Helsinki-NLP/opus-mt-en-fr",
                "facebook/mbart-large-50-many-to-many-mmt"
            ],
            "text_generation": [
                "gpt2",
                "gpt2-medium",
                "EleutherAI/gpt-neo-2.7B",
                "microsoft/DialoGPT-medium"
            ]
        },
        "computer_vision": {
            "image_classification": [
                "google/vit-base-patch16-224",
                "microsoft/resnet-50",
                "facebook/deit-base-distilled-patch16-224",
                "microsoft/beit-base-patch16-224"
            ],
            "object_detection": [
                "facebook/detr-resnet-50",
                "hustvl/yolos-tiny",
                "microsoft/table-transformer-detection"
            ],
            "image_segmentation": [
                "nvidia/segformer-b0-finetuned-ade-512-512",
                "facebook/detr-resnet-50-panoptic",
                "facebook/maskformer-swin-base-ade"
            ]
        },
        "audio": {
            "speech_recognition": [
                "facebook/wav2vec2-base-960h",
                "openai/whisper-base",
                "microsoft/wavlm-base-plus"
            ],
            "audio_classification": [
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                "facebook/wav2vec2-base",
                "superb/wav2vec2-base-superb-ks"
            ]
        },
        "multimodal": {
            "image_text": [
                "openai/clip-vit-base-patch32",
                "microsoft/layoutlmv3-base",
                "google/pix2struct-base"
            ],
            "video": [
                "MCG-NJU/videomae-base",
                "microsoft/xclip-base-patch32"
            ]
        },
        "tabular": {
            "classification": [
                "autogluon/tabular-classification",
                "microsoft/table-transformer-detection"
            ],
            "regression": [
                "autogluon/tabular-regression"
            ]
        }
    }

    @classmethod
    def get_all_models(cls) -> Dict[str, Dict[str, List[str]]]:
        """Get all available models"""
        return cls.MODELS

    @classmethod
    def get_models_by_task(cls, task: str) -> List[str]:
        """Get models for a specific task"""
        models = []
        for category in cls.MODELS.values():
            if task in category:
                models.extend(category[task])
        return models

    @classmethod
    def get_models_by_category(cls, category: str) -> Dict[str, List[str]]:
        """Get all models in a category"""
        return cls.MODELS.get(category, {})


class ModelHubEngine(BaseMLModel):
    """Engine for loading and using pretrained models"""

    def __init__(self, source: str = "huggingface"):
        super().__init__(f"model_hub_{source}", "pretrained")
        self.source = source
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.pipeline = None
        self.model_info = {}

    async def _setup(self):
        """Setup model hub connection"""
        if self.source == "huggingface":
            await self._setup_huggingface()
        elif self.source == "tensorflow_hub":
            await self._setup_tfhub()
        elif self.source == "torch_hub":
            await self._setup_torchhub()
        else:
            raise ValueError(f"Unknown model hub source: {self.source}")

    async def _setup_huggingface(self):
        """Setup Hugging Face integration"""
        try:
            from transformers import AutoModel, AutoTokenizer, pipeline
            self.hf_available = True
            logger.info("Hugging Face transformers library available")
        except ImportError:
            self.hf_available = False
            logger.warning("Hugging Face transformers not installed")

    async def _setup_tfhub(self):
        """Setup TensorFlow Hub integration"""
        try:
            import tensorflow_hub as hub
            self.tfhub_available = True
            logger.info("TensorFlow Hub available")
        except ImportError:
            self.tfhub_available = False
            logger.warning("TensorFlow Hub not installed")

    async def _setup_torchhub(self):
        """Setup PyTorch Hub integration"""
        try:
            import torch
            self.torchhub_available = True
            logger.info("PyTorch Hub available")
        except ImportError:
            self.torchhub_available = False
            logger.warning("PyTorch not installed")

    async def load_model(self, model_id: str, task: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Load a pretrained model from the hub"""
        if not self._initialized:
            await self.initialize()

        try:
            if self.source == "huggingface":
                return await self._load_huggingface_model(model_id, task, **kwargs)
            elif self.source == "tensorflow_hub":
                return await self._load_tfhub_model(model_id, **kwargs)
            elif self.source == "torch_hub":
                return await self._load_torchhub_model(model_id, **kwargs)
            else:
                raise ValueError(f"Unsupported source: {self.source}")

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    async def _load_huggingface_model(self, model_id: str, task: Optional[str] = None, **kwargs):
        """Load model from Hugging Face"""
        if not self.hf_available:
            raise ImportError("Hugging Face transformers not available")

        from transformers import (
            AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
            AutoModelForTokenClassification, AutoModelForQuestionAnswering,
            AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
        )

        # Determine model class based on task
        model_classes = {
            "text-classification": AutoModelForSequenceClassification,
            "token-classification": AutoModelForTokenClassification,
            "question-answering": AutoModelForQuestionAnswering,
            "text-generation": AutoModelForCausalLM,
            "text2text-generation": AutoModelForSeq2SeqLM,
            "default": AutoModel
        }

        model_class = model_classes.get(task, model_classes["default"])

        # Load model and tokenizer
        try:
            # Use HF token if available
            hf_token = os.getenv("HUGGINGFACE_TOKEN")

            self.model = model_class.from_pretrained(
                model_id,
                use_auth_token=hf_token if hf_token else None,
                **kwargs
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_auth_token=hf_token if hf_token else None
            )

            # Create pipeline for easy inference
            if task:
                self.pipeline = pipeline(
                    task,
                    model=self.model,
                    tokenizer=self.tokenizer
                )

            self.is_trained = True
            self.model_info = {
                "model_id": model_id,
                "task": task,
                "source": "huggingface",
                "model_type": model_class.__name__
            }

            return {
                "status": "success",
                "model_id": model_id,
                "task": task,
                "model_info": self.model_info
            }

        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {e}")
            raise

    async def _load_tfhub_model(self, model_url: str, **kwargs):
        """Load model from TensorFlow Hub"""
        if not self.tfhub_available:
            raise ImportError("TensorFlow Hub not available")

        import tensorflow_hub as hub
        import tensorflow as tf

        # Load model
        self.model = hub.load(model_url)
        self.is_trained = True

        self.model_info = {
            "model_url": model_url,
            "source": "tensorflow_hub",
            "framework": "tensorflow"
        }

        return {
            "status": "success",
            "model_url": model_url,
            "model_info": self.model_info
        }

    async def _load_torchhub_model(self, model_id: str, **kwargs):
        """Load model from PyTorch Hub"""
        if not self.torchhub_available:
            raise ImportError("PyTorch not available")

        import torch

        # Parse repo and model name
        if '/' in model_id:
            repo, model_name = model_id.split('/', 1)
        else:
            repo = "pytorch/vision"
            model_name = model_id

        # Load model
        self.model = torch.hub.load(repo, model_name, pretrained=True, **kwargs)
        self.model.eval()
        self.is_trained = True

        self.model_info = {
            "model_id": model_id,
            "repo": repo,
            "model_name": model_name,
            "source": "torch_hub",
            "framework": "pytorch"
        }

        return {
            "status": "success",
            "model_id": model_id,
            "model_info": self.model_info
        }

    async def predict(self, inputs: Any, **kwargs) -> Any:
        """Make predictions with loaded model"""
        if not self.is_trained:
            raise RuntimeError("No model loaded")

        if self.source == "huggingface":
            return await self._predict_huggingface(inputs, **kwargs)
        elif self.source == "tensorflow_hub":
            return await self._predict_tfhub(inputs, **kwargs)
        elif self.source == "torch_hub":
            return await self._predict_torchhub(inputs, **kwargs)

    async def _predict_huggingface(self, inputs: Any, **kwargs):
        """Predict using Hugging Face model"""
        if self.pipeline:
            # Use pipeline for inference
            results = self.pipeline(inputs, **kwargs)
            return results
        else:
            # Manual inference
            if self.tokenizer:
                encoded = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    **kwargs
                )

                outputs = self.model(**encoded)

                # Process outputs based on task
                if hasattr(outputs, "logits"):
                    import torch
                    predictions = torch.softmax(outputs.logits, dim=-1)
                    return predictions.tolist()
                else:
                    return outputs

    async def _predict_tfhub(self, inputs: Any, **kwargs):
        """Predict using TensorFlow Hub model"""
        import tensorflow as tf

        # Prepare inputs
        if isinstance(inputs, str):
            inputs = [inputs]

        # Run inference
        outputs = self.model(inputs)

        return outputs.numpy().tolist() if hasattr(outputs, 'numpy') else outputs

    async def _predict_torchhub(self, inputs: Any, **kwargs):
        """Predict using PyTorch Hub model"""
        import torch

        # Prepare inputs
        if hasattr(self.model, 'transform'):
            inputs = self.model.transform(inputs)

        with torch.no_grad():
            outputs = self.model(inputs)

        return outputs.numpy().tolist() if hasattr(outputs, 'numpy') else outputs.tolist()

    async def train(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """Fine-tune the pretrained model"""
        if not self.is_trained:
            raise RuntimeError("No model loaded")

        if self.source == "huggingface":
            return await self._finetune_huggingface(X, y, **kwargs)
        else:
            raise NotImplementedError(f"Fine-tuning not implemented for {self.source}")

    async def _finetune_huggingface(self, X: Any, y: Any, **kwargs):
        """Fine-tune Hugging Face model"""
        from transformers import TrainingArguments, Trainer
        import torch
        from torch.utils.data import Dataset

        class CustomDataset(Dataset):
            def __init__(self, texts, labels, tokenizer):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                )

                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                }

        # Create dataset
        dataset = CustomDataset(X, y, self.tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=kwargs.get('epochs', 3),
            per_device_train_batch_size=kwargs.get('batch_size', 16),
            per_device_eval_batch_size=kwargs.get('batch_size', 16),
            warmup_steps=kwargs.get('warmup_steps', 500),
            weight_decay=kwargs.get('weight_decay', 0.01),
            logging_dir='./logs',
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,  # Should use separate validation set
        )

        # Train
        trainer.train()

        return {
            "status": "success",
            "training_completed": True,
            "final_loss": trainer.state.log_history[-1].get('loss', None)
        }

    async def evaluate(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """Evaluate the model"""
        predictions = await self.predict(X, **kwargs)

        # Calculate metrics based on task
        from sklearn.metrics import accuracy_score, f1_score

        if self.model_info.get("task") == "text-classification":
            # Extract predicted classes
            if isinstance(predictions, list) and len(predictions) > 0:
                if isinstance(predictions[0], dict):
                    pred_classes = [p['label'] for p in predictions]
                else:
                    import numpy as np
                    pred_classes = np.argmax(predictions, axis=1)

                accuracy = accuracy_score(y, pred_classes)
                f1 = f1_score(y, pred_classes, average='weighted')

                return {
                    "accuracy": float(accuracy),
                    "f1_score": float(f1),
                    "n_samples": len(y)
                }

        return {"message": "Evaluation not implemented for this task"}

    async def execute(self, data: Any, params: Dict[str, Any]) -> Any:
        """Execute model hub operations"""
        operation = params.get("operation", "predict")

        if operation == "load":
            return await self.load_model(
                params.get("model_id"),
                params.get("task"),
                **params.get("kwargs", {})
            )
        elif operation == "predict":
            return await self.predict(data, **params.get("kwargs", {}))
        elif operation == "train":
            return await self.train(
                data,
                params.get("labels"),
                **params.get("kwargs", {})
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")

    async def list_available_models(self, task: Optional[str] = None) -> List[str]:
        """List available models for a task"""
        if task:
            return PretrainedModelRegistry.get_models_by_task(task)
        else:
            all_models = []
            for category in PretrainedModelRegistry.MODELS.values():
                for models in category.values():
                    all_models.extend(models)
            return all_models

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if self.source == "huggingface":
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                model_info = api.model_info(model_id)

                return {
                    "model_id": model_id,
                    "downloads": model_info.downloads,
                    "likes": model_info.likes,
                    "tags": model_info.tags,
                    "pipeline_tag": model_info.pipeline_tag,
                    "library_name": model_info.library_name
                }
            except:
                return {"model_id": model_id, "source": self.source}

        return {"model_id": model_id, "source": self.source}
