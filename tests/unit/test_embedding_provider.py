"""Tests for embedding providers."""

import pytest
import openai

from backend.infrastructure.retrieval.embeddings import (
    FakeEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from backend.domain.knowledge import EmbeddingProviderError


@pytest.mark.asyncio
async def test_fake_embedding_provider():
    provider = FakeEmbeddingProvider(dimensions=10)
    assert provider.provider_name == "fake"
    assert provider.model_name == "fake-embedding-model"
    assert provider.dimensions == 10

    texts = ["hello", "world"]
    embeddings = await provider.embed(texts)
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 10
    assert embeddings[0] == [0.1] * 10


@pytest.mark.asyncio
async def test_openai_embedding_provider_empty_list():
    provider = OpenAIEmbeddingProvider(api_key="fake-key")
    embeddings = await provider.embed([])
    assert embeddings == []


@pytest.mark.asyncio
async def test_openai_embedding_provider_api_error(monkeypatch):
    provider = OpenAIEmbeddingProvider(api_key="fake-key")

    async def mock_create(*args, **kwargs):
        raise openai.APIConnectionError(request=None)

    monkeypatch.setattr(provider._client.embeddings, "create", mock_create)

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await provider.embed(["test"])
    
    assert exc_info.value.error_type == "openai_api_error"


@pytest.mark.asyncio
async def test_openai_embedding_provider_success(monkeypatch):
    provider = OpenAIEmbeddingProvider(
        api_key="fake-key",
        dimensions=2,
        batch_size=2
    )

    class MockEmbedding:
        def __init__(self, index: int, embedding: list[float]):
            self.index = index
            self.embedding = embedding

    class MockResponse:
        def __init__(self, data):
            self.data = data

    async def mock_create(*args, **kwargs):
        # inputs are batched
        input_texts = kwargs["input"]
        data = [MockEmbedding(index=i, embedding=[float(i), float(i)]) for i in range(len(input_texts))]
        return MockResponse(data=data)

    monkeypatch.setattr(provider._client.embeddings, "create", mock_create)

    texts = ["one", "two", "three"]
    embeddings = await provider.embed(texts)

    assert len(embeddings) == 3
    assert len(embeddings[0]) == 2
    # First batch (size 2): index 0 -> [0.0, 0.0], index 1 -> [1.0, 1.0]
    # Second batch (size 1): index 0 -> [0.0, 0.0]
    assert embeddings[0] == [0.0, 0.0]
    assert embeddings[1] == [1.0, 1.0]
    assert embeddings[2] == [0.0, 0.0]
