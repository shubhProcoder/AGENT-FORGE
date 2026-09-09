"""Unit tests for CandidateValidator."""

from __future__ import annotations

import pytest

from backend.verification.candidate_validator import (
    CandidateValidator,
    ValidationResult,
)


class TestCandidateValidator:
    def setup_method(self):
        self.validator = CandidateValidator()

    def test_valid_python_code(self):
        """Valid Python code parses cleanly."""
        code = """def second_largest_unique(numbers):
    unique = sorted(set(numbers))
    return unique[-2] if len(unique) >= 2 else None
"""
        result = self.validator.validate(code)
        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.error_type is None
        assert result.line is None

    def test_syntax_error_unclosed_paren(self):
        """Syntax error is caught with line and column numbers."""
        code = "def foo(\n    return 42\n"
        result = self.validator.validate(code)
        assert result.valid is False
        assert result.error_type == "SYNTAX_ERROR"
        assert result.line is not None
        assert result.message is not None

    def test_syntax_error_invalid_token(self):
        """Syntax error with invalid tokens."""
        code = "def foo():\n    x = @@@\n"
        result = self.validator.validate(code)
        assert result.valid is False
        assert result.error_type == "SYNTAX_ERROR"
        assert result.line == 2

    def test_indentation_error(self):
        """Indentation error is caught as SYNTAX_ERROR."""
        code = "def foo():\nreturn 42\n"
        result = self.validator.validate(code)
        assert result.valid is False
        assert result.error_type == "SYNTAX_ERROR"

    def test_empty_source(self):
        """Empty source is caught before parsing."""
        result = self.validator.validate("")
        assert result.valid is False
        assert result.error_type == "EMPTY_SOURCE"

    def test_required_function_check(self):
        """Validator flags missing required functions in warnings."""
        validator = CandidateValidator(required_functions=["second_largest_unique"])
        code = "def other_function(): pass\n"
        result = validator.validate(code)
        assert result.valid is True
        assert len(result.warnings) == 1
        assert "second_largest_unique" in result.warnings[0]
