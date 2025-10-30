"""Answer processing and extraction utilities for GAIA benchmark evaluation."""

import re
from typing import Optional, Tuple, Dict, Any
import json


class AnswerProcessor:
    """Process and extract clean answers from agent responses."""

    # Common answer markers that indicate a final answer
    ANSWER_MARKERS = [
        r"FINAL_ANSWER_SUBMITTED:\s*(.+)",  # From FinalAnswerTool
        r"FINAL ANSWER:\s*(.+?)(?:\.|$)",
        r"final answer:\s*(.+?)(?:\.|$)",
        r"the answer is:\s*(.+?)(?:\.|$)",
        r"answer:\s*(.+?)(?:\.|$)",
        r"in conclusion:\s*(.+?)(?:\.|$)",
        r"therefore:\s*(.+?)(?:\.|$)",
        r"<final_answer>\s*(.+?)\s*</final_answer>",
        r"\[final answer\]\s*(.+?)(?:\.|$)",
    ]

    @classmethod
    def extract_answer(cls, text: str) -> Tuple[str, Dict[str, Any]]:
        """Extract the final answer from agent response text.

        Args:
            text: The raw agent response text

        Returns:
            Tuple of (extracted_answer, metadata_dict)
            metadata_dict contains:
                - extraction_method: How the answer was extracted
                - confidence: Confidence score (0-1)
                - raw_text: Original text
        """
        if not text:
            return "", {"extraction_method": "empty", "confidence": 0.0, "raw_text": text}

        # First check if this is from FinalAnswerTool
        if "FINAL_ANSWER_SUBMITTED:" in text:
            match = re.search(r"FINAL_ANSWER_SUBMITTED:\s*(.+)", text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                return answer, {
                    "extraction_method": "final_answer_tool",
                    "confidence": 1.0,
                    "raw_text": text
                }

        # Try pattern matching with answer markers
        for pattern in cls.ANSWER_MARKERS:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # Clean the extracted answer
                answer = cls._clean_answer(answer)
                if answer:  # Only return if we got a non-empty answer
                    return answer, {
                        "extraction_method": "pattern_matching",
                        "pattern": pattern,
                        "confidence": 0.9,
                        "raw_text": text
                    }

        # Fallback: Look for the last substantive sentence/phrase
        answer = cls._extract_fallback_answer(text)
        if answer:
            return answer, {
                "extraction_method": "fallback",
                "confidence": 0.5,
                "raw_text": text
            }

        # Last resort: return cleaned full text
        cleaned_text = cls._clean_answer(text)
        return cleaned_text, {
            "extraction_method": "full_text",
            "confidence": 0.3,
            "raw_text": text
        }

    @classmethod
    def _clean_answer(cls, answer: str) -> str:
        """Clean and normalize an answer string.

        Args:
            answer: Raw answer string

        Returns:
            Cleaned answer string
        """
        if not answer:
            return ""

        # Remove leading/trailing whitespace
        answer = answer.strip()

        # Remove markdown formatting
        answer = re.sub(r'\*\*(.+?)\*\*', r'\1', answer)  # Bold **text**
        answer = re.sub(r'__(.+?)__', r'\1', answer)  # Bold __text__
        answer = re.sub(r'\*(.+?)\*', r'\1', answer)  # Italic *text*
        answer = re.sub(r'_(.+?)_', r'\1', answer)  # Italic _text_
        answer = re.sub(r'`(.+?)`', r'\1', answer)  # Code `text`
        answer = re.sub(r'```[^\n]*\n?(.*?)```', r'\1', answer, flags=re.DOTALL)  # Code blocks

        # Remove common prefixes if they still exist
        prefixes = [
            "final answer:",
            "the answer is:",
            "answer:",
            "therefore:",
            "in conclusion:",
            "thus:",
            "hence:",
        ]

        lower_answer = answer.lower()
        for prefix in prefixes:
            if lower_answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
                lower_answer = answer.lower()

        # Remove quotes if they wrap the entire answer
        if len(answer) > 2:
            if (answer.startswith('"') and answer.endswith('"')) or \
               (answer.startswith("'") and answer.endswith("'")):
                answer = answer[1:-1].strip()

        # Remove trailing punctuation that's not part of the answer
        # But keep punctuation that might be part of titles or proper answers
        if answer.endswith('.') and not re.search(r'[A-Z]\.$', answer):  # Not an abbreviation
            answer = answer[:-1].strip()

        # Handle "let me..." or similar meta-text
        meta_patterns = [
            r"^let me.*?final answer[:\s]+(.+)",
            r"^i.*?final answer[:\s]+(.+)",
            r"^the.*?final answer[:\s]+(.+)",
        ]

        for pattern in meta_patterns:
            match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()

        # Clean up any remaining formatting issues
        answer = re.sub(r'\s+', ' ', answer)  # Normalize whitespace
        answer = answer.strip()

        return answer

    @classmethod
    def _extract_fallback_answer(cls, text: str) -> str:
        """Extract answer using fallback heuristics.

        Args:
            text: The full response text

        Returns:
            Extracted answer or empty string
        """
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)

        # Look for sentences that might contain answers
        for sent in reversed(sentences):  # Start from the end
            sent = sent.strip()

            # Skip meta-commentary
            if any(phrase in sent.lower() for phrase in [
                "let me", "i need", "i should", "i will", "i can",
                "updating", "checking", "looking", "searching"
            ]):
                continue

            # Skip very short sentences
            if len(sent) < 3:
                continue

            # If it looks like an answer (contains a number, name, or short phrase)
            if re.search(r'\d+', sent) or len(sent) < 100:
                cleaned = cls._clean_answer(sent)
                if cleaned and cleaned != text:  # Make sure we're not returning the full text
                    return cleaned

        return ""

    @classmethod
    def normalize_for_comparison(cls, text: str) -> str:
        """Normalize text for comparison (used in scoring).

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower().strip()

        # Remove punctuation except hyphens and apostrophes (which might be part of answers)
        text = re.sub(r'[^\w\s\-\']', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Handle common number formats
        text = re.sub(r'(\d+),(\d+)', r'\1\2', text)  # Remove commas from numbers

        return text

    @classmethod
    def process_agent_response(cls, response: str) -> Dict[str, Any]:
        """Process a complete agent response and extract structured information.

        Args:
            response: Raw agent response

        Returns:
            Dictionary containing:
                - answer: The extracted answer
                - raw_response: Original response
                - metadata: Extraction metadata
        """
        answer, metadata = cls.extract_answer(response)

        # Additional validation
        if answer:
            # Check if answer seems incomplete
            if any(marker in answer.lower() for marker in ["...", "etc.", "and so on"]):
                metadata["warning"] = "Answer may be incomplete"
                metadata["confidence"] *= 0.8

            # Check if answer is too long (might be grabbing too much)
            if len(answer) > 500:
                metadata["warning"] = "Answer unusually long, may need manual review"
                metadata["confidence"] *= 0.9

        return {
            "answer": answer,
            "raw_response": response,
            "metadata": metadata
        }