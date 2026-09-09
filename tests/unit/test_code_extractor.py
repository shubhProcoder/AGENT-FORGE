"""Unit tests for CodeExtractionService."""

from __future__ import annotations

import pytest

from backend.verification.code_extractor import (
    CandidateCode,
    CodeExtractionError,
    CodeExtractionService,
)


class TestCodeExtractionService:
    def setup_method(self):
        self.extractor = CodeExtractionService()

    def test_markdown_fence_with_python_tag(self):
        """Extracts code enclosed in ```python ... ```."""
        text = """Here is the solution:
```python
def second_largest_unique(numbers):
    unique = sorted(set(numbers))
    return unique[-2] if len(unique) >= 2 else None
```
Hope that helps!"""
        result = self.extractor.extract(text)
        assert isinstance(result, CandidateCode)
        assert result.language == "python"
        assert result.extracted_from == "markdown_fence"
        assert "def second_largest_unique(numbers):" in result.source
        assert "Hope that helps!" not in result.source

    def test_markdown_fence_without_tag(self):
        """Extracts code enclosed in untagged ``` ... ```."""
        text = """```
def add(a, b):
    return a + b
```"""
        result = self.extractor.extract(text)
        assert "def add(a, b):" in result.source

    def test_raw_python_extraction(self):
        """Extracts plain Python code when no markdown fence is provided."""
        text = """def second_largest_unique(numbers):
    unique = sorted(set(numbers))
    return unique[-2] if len(unique) >= 2 else None"""
        result = self.extractor.extract(text)
        assert result.extracted_from == "raw_text"
        assert "def second_largest_unique(numbers):" in result.source
        assert len(result.warnings) > 0

    def test_empty_response_rejected(self):
        """Rejects None, empty, or whitespace-only strings."""
        with pytest.raises(CodeExtractionError, match="Empty response"):
            self.extractor.extract("")

        with pytest.raises(CodeExtractionError, match="Empty response"):
            self.extractor.extract("   \n\t  ")

        with pytest.raises(CodeExtractionError, match="Empty response"):
            self.extractor.extract(None)

    def test_unsupported_language_rejected(self):
        """Rejects code blocks in unsupported languages like JavaScript or Rust."""
        text = """```javascript
function solution(numbers) {
    return numbers[0];
}
```"""
        with pytest.raises(CodeExtractionError, match="Unsupported language 'javascript'"):
            self.extractor.extract(text)

    def test_ambiguous_multiple_code_blocks_rejected(self):
        """Rejects responses with multiple ambiguous Python blocks."""
        text = """Here is option A:
```python
def solution(nums):
    return 1
```
And here is option B:
```python
def solution(nums):
    return 2
```"""
        with pytest.raises(CodeExtractionError, match="Ambiguous response"):
            self.extractor.extract(text)

    def test_conversational_non_code_rejected(self):
        """Rejects conversational prose that does not contain code."""
        text = "I cannot solve this problem because I need more information about the requirements."
        with pytest.raises(CodeExtractionError, match="No executable Python code found"):
            self.extractor.extract(text)
