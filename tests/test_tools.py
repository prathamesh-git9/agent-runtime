from __future__ import annotations

import pytest

from agent_runtime.errors import ToolNotFound
from agent_runtime.tools import Approval, Tool, ToolRegistry


def test_register_and_get_tool():
    registry = ToolRegistry()

    def handler() -> str:
        return "ok"

    tool = Tool(name="ping", description="Ping.", handler=handler)
    registered = registry.register(tool)

    assert registered is tool
    assert registry.get("ping") is tool


def test_tool_decorator_registers_and_preserves_function():
    registry = ToolRegistry()

    @registry.tool("ping", description="Ping.")
    def handler() -> str:
        return "ok"

    assert registry.get("ping").handler is handler
    assert handler() == "ok"


def test_get_unknown_tool_raises_tool_not_found():
    registry = ToolRegistry()

    with pytest.raises(ToolNotFound):
        registry.get("missing")


def test_contains_reports_registered_names():
    registry = ToolRegistry()
    registry.register(Tool(name="ping", description="Ping.", handler=lambda: "ok"))

    assert "ping" in registry
    assert "missing" not in registry


def test_specs_include_public_tool_metadata():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="refund",
            description="Issue a refund.",
            handler=lambda amount: f"refunded {amount}",
            approval=Approval.REQUIRED,
            parameters={"amount": "number"},
        )
    )

    assert registry.specs() == [
        {
            "name": "refund",
            "description": "Issue a refund.",
            "parameters": {"amount": "number"},
            "approval": "required",
        }
    ]


def test_decorator_uses_docstring_description_when_omitted():
    registry = ToolRegistry()

    @registry.tool("lookup")
    def lookup() -> str:
        """Look up a record."""
        return "record"

    assert registry.get("lookup").description == "Look up a record."
