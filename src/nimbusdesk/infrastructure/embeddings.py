"""FastEmbed adapter — the concrete Embedder behind the port in rag/ports.py.

FastEmbed runs quantized ONNX models on CPU: no GPU, no API key, no per-token
cost. The model file (~65 MB) is downloaded on first use and cached locally.

Note the asymmetry handled here: `query_embed` prepends the instruction prefix
bge models were trained with for queries, while `passage_embed` embeds
documents plainly. Using the same call for both is a silent retrieval-quality
bug — which is why the port forces the distinction.
"""

from typing import Sequence

from fastembed import TextEmbedding


class FastEmbedEmbedder:
    def __init__(self, model_name: str, dimension: int) -> None:
        self._model = TextEmbedding(model_name=model_name)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed([text]))).tolist()
