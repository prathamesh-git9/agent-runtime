from __future__ import annotations

from agent_runtime.api import create_app
from agent_runtime.journal import Journal
from agent_runtime.llm import Finish, ScriptedPlanner
from agent_runtime.tools import ToolRegistry

registry = ToolRegistry()


@registry.tool("echo", description="Echo a message.", parameters={"message": "string"})
def echo(message: str = "") -> str:
    return message


journal = Journal()
planner = ScriptedPlanner([Finish(output="demo runtime is ready")])
app = create_app(journal, planner, registry)
