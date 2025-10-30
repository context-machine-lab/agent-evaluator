"""Final Answer Tool for agents to submit clean, validated answers."""

import re
from typing import Any, Dict, Optional
from ..core.registry import register_tool


@register_tool("final_answer")
async def final_answer_tool(answer: str, **kwargs) -> str:
    """Submit a final answer to the question.

    This tool should be used when you have determined the final answer to the question.
    Provide ONLY the answer value without any prefixes, formatting, or explanations.

    Args:
        answer: The final answer to submit. Should be clean and concise.
                Examples:
                - For numeric answers: "42" or "17.5"
                - For text answers: "Paris" or "The Eiffel Tower"
                - For dates: "04/15/18" or "January 1, 2023"
                - For lists: "apple, banana, orange"

    Returns:
        Confirmation message with the submitted answer
    """
    # Clean the answer - remove common formatting issues
    cleaned_answer = answer.strip()

    # Remove markdown formatting
    cleaned_answer = re.sub(r'\*\*(.+?)\*\*', r'\1', cleaned_answer)  # Bold
    cleaned_answer = re.sub(r'__(.+?)__', r'\1', cleaned_answer)  # Bold
    cleaned_answer = re.sub(r'\*(.+?)\*', r'\1', cleaned_answer)  # Italic
    cleaned_answer = re.sub(r'_(.+?)_', r'\1', cleaned_answer)  # Italic
    cleaned_answer = re.sub(r'`(.+?)`', r'\1', cleaned_answer)  # Code

    # Remove common prefixes if they somehow got included
    prefixes_to_remove = [
        "final answer:",
        "final answer is",
        "the answer is:",
        "answer:",
        "in conclusion:",
        "therefore:",
    ]

    lower_cleaned = cleaned_answer.lower()
    for prefix in prefixes_to_remove:
        if lower_cleaned.startswith(prefix):
            cleaned_answer = cleaned_answer[len(prefix):].strip()
            break

    # Remove quotes if they wrap the entire answer
    if len(cleaned_answer) > 2:
        if (cleaned_answer.startswith('"') and cleaned_answer.endswith('"')) or \
           (cleaned_answer.startswith("'") and cleaned_answer.endswith("'")):
            cleaned_answer = cleaned_answer[1:-1].strip()

    # Store the cleaned answer in a structured format
    result = {
        "type": "final_answer",
        "answer": cleaned_answer,
        "raw_answer": answer
    }

    # Return a formatted response that indicates this is a final answer
    return f"FINAL_ANSWER_SUBMITTED: {cleaned_answer}"


@register_tool("validate_answer")
async def validate_answer_tool(answer: str, question_type: Optional[str] = None, **kwargs) -> str:
    """Validate and format an answer before submission.

    Args:
        answer: The answer to validate
        question_type: Optional hint about the expected answer type
                      (e.g., "number", "date", "text", "list")

    Returns:
        Validation result and formatted answer
    """
    validation_issues = []
    formatted_answer = answer.strip()

    # Check if answer is empty
    if not formatted_answer:
        validation_issues.append("Answer is empty")
        return "VALIDATION_FAILED: Answer cannot be empty"

    # Type-specific validation
    if question_type:
        if question_type == "number":
            # Check if it's a valid number
            try:
                # Remove commas from numbers
                test_num = formatted_answer.replace(",", "")
                float(test_num)
                formatted_answer = test_num
            except ValueError:
                validation_issues.append(f"Expected a number but got: {formatted_answer}")

        elif question_type == "date":
            # Common date formats
            date_patterns = [
                r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YY or MM/DD/YYYY
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YY or MM-DD-YYYY
                r'[A-Za-z]+ \d{1,2}, \d{4}',  # Month DD, YYYY
            ]

            if not any(re.match(pattern, formatted_answer) for pattern in date_patterns):
                validation_issues.append(f"Answer doesn't appear to be a valid date: {formatted_answer}")

        elif question_type == "yes_no":
            lower_answer = formatted_answer.lower()
            if lower_answer not in ["yes", "no", "true", "false"]:
                validation_issues.append(f"Expected yes/no answer but got: {formatted_answer}")
            # Normalize to yes/no
            if lower_answer in ["true", "yes"]:
                formatted_answer = "yes"
            elif lower_answer in ["false", "no"]:
                formatted_answer = "no"

    # Check for common issues
    if "..." in formatted_answer or "etc." in formatted_answer:
        validation_issues.append("Answer appears to be incomplete (contains '...' or 'etc.')")

    if len(formatted_answer) > 500:
        validation_issues.append(f"Answer is very long ({len(formatted_answer)} chars). Consider being more concise.")

    if validation_issues:
        return f"VALIDATION_WARNING: {'; '.join(validation_issues)}\nFormatted answer: {formatted_answer}"
    else:
        return f"VALIDATION_PASSED: {formatted_answer}"