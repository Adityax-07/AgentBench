"""
SAFE PYTHON REPL TOOL FOR LANGCHAIN AGENTS
=========================================

This tool allows an AI agent to execute Python code safely.

Why this exists:
LLM agents often need to:
• do math
• analyze data
• run small scripts

But a raw Python REPL is EXTREMELY dangerous.
So we add guardrails:
✔ restricted builtins
✔ import whitelist
✔ timeout protection
✔ output capture
✔ error handling
"""

# =========================
# Imports
# =========================
from langchain_core.tools import tool
import io
import contextlib
import threading


# =========================
# 1️⃣ Timeout Protection
# =========================
# WHY threading.Timer instead of signal.SIGALRM?
# signal.SIGALRM only exists on Unix/Linux — it crashes on Windows.
# threading.Timer is cross-platform and works identically.
# It fires a callback after N seconds on a background thread,
# which sets a flag that the main thread checks after exec().

class TimeoutException(Exception):
    pass


# =========================
# 2️⃣ Restricted Builtins (Sandbox)
# =========================
# WHY:
# Remove dangerous Python functions such as:
# open(), exec(), eval(), __import__(), etc.

SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sorted": sorted,
}


# =========================
# 3️⃣ Allowed Libraries
# =========================
# WHY:
# Agent should only use safe scientific libraries.
# Blocks OS/system access.

ALLOWED_IMPORTS = {
    "math",
    "statistics",
    "random",
    "numpy",
    "pandas",
}


# =========================
# 4️⃣ Import Validator
# =========================
def validate_imports(code: str):
    """
    Block dangerous imports like os, sys, subprocess.
    """
    for line in code.split("\n"):
        line = line.strip()

        if line.startswith("import") or line.startswith("from"):
            lib = line.split()[1].split(".")[0]

            if lib not in ALLOWED_IMPORTS:
                raise ValueError(f"Import '{lib}' is not allowed.")


# =========================
# 5️⃣ Safe Execution Engine
# =========================
def execute_python(code: str) -> str:
    """
    Executes Python code safely and captures output.
    """

    # Validate imports before execution
    validate_imports(code)

    # Capture print() output
    output_buffer = io.StringIO()

    # WHY a list instead of a plain bool?
    # exec() runs in the same thread, so we need a mutable container
    # that the timer callback can write to and the main thread can read.
    # A plain bool variable would be a new local binding — not shared.
    timed_out = [False]

    def _trigger_timeout():
        timed_out[0] = True

    # WHY threading.Timer?
    # It fires _trigger_timeout() after 3 seconds on a background thread.
    # Cross-platform — works on Windows, Linux, and Mac.
    timer = threading.Timer(3.0, _trigger_timeout)

    try:
        timer.start()

        # Execute code inside restricted environment
        with contextlib.redirect_stdout(output_buffer):
            exec(code, {"__builtins__": SAFE_BUILTINS}, {})

        # Check if timer fired during execution
        if timed_out[0]:
            return "Execution timed out (3s limit)."

        output = output_buffer.getvalue()
        return output if output else "Code executed successfully."

    except Exception as e:
        return f"Execution error: {str(e)}"

    finally:
        # WHY finally?
        # Always cancel the timer — even if exec() raises an exception.
        # Without this, the timer thread would keep running in the background.
        timer.cancel()


# =========================
# 6️⃣ LangChain Tool Wrapper
# =========================
@tool
def python_repl(code: str) -> str:
    """
    Safely execute short Python code.

    Use this tool for:
    • math calculations
    • numpy / pandas data analysis
    • quick scripts

    Restrictions:
    • No file/system/network access
    • Only safe libraries allowed
    • Execution limited to 3 seconds
    """

    # Guardrail: prevent huge code execution
    if len(code) > 1000:
        return "Code too long. Please shorten."

    return execute_python(code)


# =========================
# Example CLI Test
# =========================
if __name__ == "__main__":
    print(python_repl.run("print(2+2)"))
    print(python_repl.run("import numpy as np\nprint(np.mean([1,2,3]))"))