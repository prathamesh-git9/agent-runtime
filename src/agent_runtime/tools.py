"""Tool registry and approval policy.

A tool declares whether it is *safe to replay*. That flag is the crux of durable
execution: replaying `read_file` costs nothing, but replaying `send_email` sends
a second email. Effectful tools must therefore have their results served from
the journal on resume and never re-invoked — see `runtime.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agent_runtime.errors import ToolNotFound


class Approval(StrEnum):
    AUTO = "auto"          # run without asking
    REQUIRED = "required"  # suspend the run until a human decides


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    approval: Approval = Approval.AUTO
    # False for anything with a side effect the world can observe. Effectful
    # tools are still durable — their recorded result is replayed — but they
    # must never be re-executed to reconstruct state.
    idempotent: bool = True
    parameters: dict[str, str] = field(default_factory=dict)

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "approval": str(self.approval),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def tool(
        self,
        name: str,
        description: str = "",
        *,
        approval: Approval = Approval.AUTO,
        idempotent: bool = True,
        parameters: dict[str, str] | None = None,
    ) -> Callable:
        """Decorator form: @registry.tool("send_email", approval=REQUIRED)."""

        def decorator(fn: Callable) -> Callable:
            self.register(
                Tool(
                    name=name,
                    description=description or (fn.__doc__ or "").strip(),
                    handler=fn,
                    approval=approval,
                    idempotent=idempotent,
                    parameters=parameters or {},
                )
            )
            return fn

        return decorator

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFound(f"no tool registered as '{name}'") from exc

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)
