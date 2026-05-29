"""Abstract base class for tools callable by the Agent."""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """A tool that the Agent can invoke via function calling.

    Subclasses must implement ``name``, ``description``, ``parameters``,
    and ``execute``.  The ``description`` and ``name`` exposed to the LLM
    can be overridden at runtime (e.g. from YAML config) without modifying
    the tool source code — call ``override_description()`` after construction.
    """

    # Runtime overrides — set by tool_factory when YAML config provides them.
    _description_override: str | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Default description shown to the LLM.

        Override at runtime via ``override_description(text)`` to use a
        YAML-configured value instead of the hardcoded default.
        """

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for the tool's parameters."""

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool and return a string result."""
        ...

    # ── Runtime override helpers ──────────────────────────────────────────

    def override_description(self, text: str) -> None:
        """Replace the default description with a YAML-configured value.

        The overridden text is what the LLM receives in ``to_tool_schema()``,
        allowing prompt tuning without touching Python source.
        """
        self._description_override = text.strip() or None

    def effective_description(self) -> str:
        """Return the description the LLM will see (override preferred)."""
        return self._description_override or self.description

    def to_tool_schema(self) -> dict:
        """Return the tool definition as a function-calling schema."""
        return {
            "name": self.name,
            "description": self.effective_description(),
            "parameters": self.parameters,
        }
