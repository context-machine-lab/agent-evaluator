"""Python interpreter tool for secure code execution."""

from agent1.tools.executor.local_python_executor import (
    BASE_BUILTIN_MODULES,
    BASE_PYTHON_TOOLS,
    evaluate_python_code,
)
from agent1.registry import register_tool


@register_tool("python_interpreter")
async def python_interpreter(code: str, **kwargs) -> str:
    """
    Evaluate Python code in a secure sandboxed environment.

    This tool evaluates python code and can be used to perform calculations,
    data processing, and other computational tasks.

    Args:
        code: The Python code to run in the interpreter.
        **kwargs: Additional arguments (ignored, for compatibility).

    Returns:
        A string containing the stdout output and the final result of execution.

    Example:
        >>> await python_interpreter("print('Hello'); 2 + 2")
        "Stdout:\\nHello\\nOutput: 4"
    """
    authorized_imports = list(set(BASE_BUILTIN_MODULES))

    try:
        state = {}
        output, is_final_answer = evaluate_python_code(
            code,
            state=state,
            static_tools=BASE_PYTHON_TOOLS.copy(),
            custom_tools={},
            authorized_imports=authorized_imports,
        )

        stdout = str(state.get('_print_outputs', ''))
        result_str = f"Stdout:\n{stdout}\nOutput: {output}"

        return result_str

    except Exception as e:
        return f"Error executing Python code: {str(e)}"


__all__ = ["python_interpreter"]
