import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.config import config
from app.llm import ChatOpenAI
from app.schema import Step
from app.tool.base import ToolCollection


class Memory(BaseModel):
    """A memory for an agent."""

    messages: List[Dict[str, Any]] = Field(default_factory=list)
    user_message: Optional[str] = None

    def add_message(self, role: str, content: str):
        """Add a message to the memory."""
        self.messages.append({"role": role, "content": content})

    def clear(self):
        """Clear the memory."""
        self.messages = []
        self.user_message = None

    def __str__(self):
        """Return a string representation of the memory."""
        return "\n".join(
            [f"{message['role']}: {message['content']}" for message in self.messages]
        )


class Agent(ABC):
    """An abstract base class for agents."""

    name: str
    description: str
    system_prompt: str
    next_step_prompt: str

    max_steps: int = 10
    max_observe: int = 10000

    available_tools: ToolCollection

    def __init__(self, **kwargs):
        self.memory = Memory()
        self.steps: List[Step] = []

    @abstractmethod
    async def run(self) -> str:
        """Run the agent."""
        pass

    async def observe(self, observation: str) -> str:
        """Observe the environment."""
        # Truncate the observation if it's too long
        if len(observation) > self.max_observe:
            observation = observation[: self.max_observe] + "..."

        self.memory.add_message("system", observation)
        return observation

    async def _take_next_step(self) -> Optional[Step]:
        """Take the next step in the agent's plan."""
        # Construct the prompt
        prompt = self.next_step_prompt.format(
            steps="\n".join([str(step) for step in self.steps]),
            tool_docs=self.available_tools.to_str(),
            user_message=self.memory.user_message,
        )

        # Call the LLM
        llm_response = await ChatOpenAI.call(
            prompt,
            stop=["\nTOOL", "\nUSER", "\nASSISTANT", "\nOBSERVATION"],
            max_tokens=1024,
        )

        # Parse the LLM response
        try:
            step = Step.from_llm_response(llm_response)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"LLM response: {llm_response}")
            return None

        return step

    async def think(self) -> bool:
        """Think about the next step."""
        self.memory.clear()

        # Add the system prompt to the memory
        self.memory.add_message("system", self.system_prompt)

        # Take steps until a terminal state is reached
        for _ in range(self.max_steps):
            step = await self._take_next_step()
            if step is None:
                break

            self.steps.append(step)

            if step.tool == "Terminate":
                break

        # Return whether the agent reached a terminal state
        return self.steps[-1].tool == "Terminate"

    async def chat(self, user_message: str) -> str:
        """Chat with the user."""
        self.memory.user_message = user_message
        self.memory.add_message("user", user_message)

        await self.think()

        # Return the result
        return self.steps[-1].result 