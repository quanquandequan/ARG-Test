"""Abstract base class for tools callable by the Agent."""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """A tool that the Agent can invoke via function calling."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for the tool's parameters."""

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool and return a string result."""
        ...

    def to_openai_schema(self) -> dict:
        """Return the tool definition in OpenAI function-calling format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
