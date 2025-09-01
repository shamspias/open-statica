from dataclasses import dataclass
from typing import List


@dataclass
class HubModel:
    id: str
    name: str
    task: str
    source: str = "huggingface"


class ModelHub:
    """
    Placeholder model hub client. Replace with real SDK calls (e.g., huggingface_hub).
    """

    def __init__(self, source: str = "huggingface"):
        self.source = source

    def search(self, query: str) -> List[HubModel]:
        seed = [
            HubModel(id="bert-base-uncased", name="BERT Base Uncased", task="text-classification"),
            HubModel(id="distilbert-base-uncased", name="DistilBERT Base", task="text-classification"),
            HubModel(id="resnet50", name="ResNet-50", task="image-classification"),
        ]
        q = query.lower()
        return [m for m in seed if q in m.id.lower() or q in m.name.lower()]
