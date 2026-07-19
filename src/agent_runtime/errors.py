"""Error taxonomy."""

from __future__ import annotations


class AgentRuntimeError(Exception):
    """Base class."""


class ConcurrentAppend(AgentRuntimeError):
    """Two workers tried to advance the same run. The loser must back off."""


class RunNotFound(AgentRuntimeError):
    pass


class ToolNotFound(AgentRuntimeError):
    pass


class ToolExecutionError(AgentRuntimeError):
    """Raised by a tool. Retryable, and each attempt is journalled."""


class ApprovalDenied(AgentRuntimeError):
    """A human refused a gated tool call. Terminal for that call, not the run."""


class MaxStepsExceeded(AgentRuntimeError):
    """The agent loop hit its step budget without producing a final answer."""


class ReplayDivergence(AgentRuntimeError):
    """The journal and the live code disagree.

    This means the agent's logic changed underneath a suspended run — a tool was
    renamed, or step ordering shifted. Failing loudly is the only safe response:
    silently continuing would apply a recorded result to the wrong call.
    """
