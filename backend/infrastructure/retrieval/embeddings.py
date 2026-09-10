"""Embedding providers for knowledge retrieval."""

import asyncio
from typing import Any

from backend.domain.knowledge import EmbeddingProvider, EmbeddingProviderError


class FakeEmbeddingProvider:
    """A fake embedding provider for testing."""

    def __init__(self, dimensions: int = 1536) -> None:
        self._dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Return a zero vector for each text, simulating API latency
        await asyncio.sleep(0.01)
        return [[0.1] * self._dimensions for _ in texts]

    @property
    def model_name(self) -> str:
        return "fake-embedding-model"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provider_name(self) -> str:
        return "fake"


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 100,
        timeout: float = 30.0,
    ) -> None:
        import openai

        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        import openai

        all_embeddings: list[list[float]] = []
        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                response = await self._client.embeddings.create(
                    input=batch,
                    model=self._model,
                    dimensions=self._dimensions,
                )
                # Sort the returned embeddings by index just to be safe
                sorted_data = sorted(response.data, key=lambda x: x.index)
                batch_embeddings = [data.embedding for data in sorted_data]
                all_embeddings.extend(batch_embeddings)
            except openai.OpenAIError as e:
                raise EmbeddingProviderError(
                    f"OpenAI embedding failed: {e}", error_type="openai_api_error"
                ) from e
            except Exception as e:
                raise EmbeddingProviderError(
                    f"Unexpected error during embedding: {e}", error_type="unexpected_error"
                ) from e

        return all_embeddings

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provider_name(self) -> str:
        return "openai"
