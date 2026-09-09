"""Candidate validator for static pre-flight syntax and structural checks.

Uses Python's abstract syntax tree (ast) to detect syntax and indentation errors
before launching any sandbox containers, preventing wasteful or unnecessary execution.
Does NOT execute code.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Outcome of static validation on candidate source code."""

    valid: bool
    error_type: str | None = None
    message: str | None = None
    line: int | None = None
    column: int | None = None
    warnings: list[str] = field(default_factory=list)


class CandidateValidator:
    """Performs static syntax and structural analysis on candidate code."""

    def __init__(self, required_functions: list[str] | None = None) -> None:
        self.required_functions = required_functions or []

    def validate(self, source_code: str) -> ValidationResult:
        """Parse source code with ast.parse and perform static checks.

        Returns:
            ValidationResult with valid=True if code parses cleanly,
            or valid=False with line/column/error_type details on syntax failure.
        """
        if not source_code or not source_code.strip():
            return ValidationResult(
                valid=False,
                error_type="EMPTY_SOURCE",
                message="Source code is empty",
            )

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                error_type="SYNTAX_ERROR",
                message=str(e.msg) if hasattr(e, "msg") else str(e),
                line=e.lineno,
                column=e.offset,
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                error_type="PARSE_ERROR",
                message=str(e),
            )

        # Check for required function definitions if specified
        warnings: list[str] = []
        if self.required_functions:
            defined_funcs = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for required_func in self.required_functions:
                if required_func not in defined_funcs:
                    warnings.append(f"Expected function '{required_func}' not found in top-level definitions")

        return ValidationResult(
            valid=True,
            warnings=warnings,
        )
