"""Code extraction service.

Responsible strictly for extracting candidate source code from model outputs,
normalizing markdown code blocks, validating supported languages, and rejecting
empty or ambiguous candidates.
Does NOT execute code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


class CodeExtractionError(Exception):
    """Raised when code extraction fails or output is invalid/ambiguous."""


@dataclass
class CandidateCode:
    """Typed container for extracted candidate source code."""

    source: str
    language: str = "python"
    extracted_from: str = "markdown_fence"  # "markdown_fence", "raw_text", "fallback"
    warnings: list[str] = field(default_factory=list)


# Regex to find markdown code fences with optional language tag
CODE_BLOCK_REGEX = re.compile(
    r"```([a-zA-Z0-9_\-\+]*)\s*\n(.*?)```",
    re.DOTALL,
)


class CodeExtractionService:
    """Extracts candidate code from LLM text responses."""

    def __init__(self, supported_languages: set[str] | None = None) -> None:
        self.supported_languages = supported_languages or {"python", "py"}

    def extract(self, response_text: str | None) -> CandidateCode:
        """Extract candidate code from raw model output.

        Raises:
            CodeExtractionError: If text is empty, no valid code exists, language is unsupported,
                                 or multiple conflicting code blocks exist.
        """
        if not response_text or not response_text.strip():
            raise CodeExtractionError("Empty response provided; cannot extract candidate code")

        text = response_text.strip()

        # Find all markdown code blocks
        blocks = CODE_BLOCK_REGEX.findall(text)

        if len(blocks) == 1:
            lang, code = blocks[0]
            clean_lang = lang.strip().lower()

            if clean_lang and clean_lang not in self.supported_languages:
                raise CodeExtractionError(
                    f"Unsupported language '{clean_lang}'. Expected one of {sorted(self.supported_languages)}"
                )

            clean_code = code.strip()
            if not clean_code:
                raise CodeExtractionError("Code block is empty")

            return CandidateCode(
                source=clean_code,
                language="python",
                extracted_from="markdown_fence",
            )

        elif len(blocks) > 1:
            # Check if one of the blocks is explicitly python and others are explanations or tests
            python_blocks = [
                (lang.strip().lower(), code.strip())
                for lang, code in blocks
                if lang.strip().lower() in self.supported_languages
            ]

            if len(python_blocks) == 1:
                clean_lang, clean_code = python_blocks[0]
                if not clean_code:
                    raise CodeExtractionError("Python code block is empty")
                return CandidateCode(
                    source=clean_code,
                    language="python",
                    extracted_from="markdown_fence",
                    warnings=["Multiple code blocks found; selected single explicit python block"],
                )
            elif len(python_blocks) > 1:
                # Ambiguous multiple candidate blocks
                raise CodeExtractionError(
                    f"Ambiguous response: found {len(python_blocks)} conflicting Python code blocks"
                )
            else:
                # Multiple blocks but none are python
                unsupported_langs = [b[0].strip().lower() for b in blocks if b[0].strip()]
                raise CodeExtractionError(
                    f"No Python code block found. Found unsupported blocks: {unsupported_langs}"
                )

        # No markdown fence found. Check if raw text looks like Python code.
        # Heuristic: must contain python keywords (def, class, import, return, etc.)
        # and not merely conversational English.
        if self._is_likely_python(text):
            return CandidateCode(
                source=text,
                language="python",
                extracted_from="raw_text",
                warnings=["Code extracted from raw text without markdown code fence"],
            )

        raise CodeExtractionError(
            "No executable Python code found in response; output appears to be natural language"
        )

    def _is_likely_python(self, text: str) -> bool:
        """Check whether raw text appears to be Python code rather than prose."""
        keywords = ["def ", "class ", "import ", "return ", "from ", "lambda "]
        has_keywords = any(kw in text for kw in keywords)

        # Check for conversational English indicators without code structure
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not has_keywords and len(sentences) > 1 and " " in sentences[0]:
            return False

        return has_keywords
