import pytest
from core.speculative import LiquidRefinementEngine, SpeculativeRacingEngine
from core.types import ToolResult


def test_liquid_refinement_error_formatting():
    # When result is not error, should return plain content
    ok_res = ToolResult(tool_name="run_command", content="All tests passed", is_error=False)
    refined_ok = LiquidRefinementEngine.inspect_and_refine(ok_res, current_iteration=1, max_iterations=10)
    assert refined_ok == "All tests passed"

    # When result is an error, should wrap with autonomous diagnostic directive
    err_res = ToolResult(
        tool_name="run_command",
        content="ModuleNotFoundError: No module named 'scipy'",
        is_error=True
    )
    refined_err = LiquidRefinementEngine.inspect_and_refine(err_res, current_iteration=2, max_iterations=10)
    assert "[TOOL FAILURE DIAGNOSTIC: 'run_command']" in refined_err
    assert "ModuleNotFoundError" in refined_err
    assert "AUTONOMOUS SELF-HEALING DIRECTIVE" in refined_err
    assert "8 iterations remaining" in refined_err
