from abc import ABC, abstractmethod
from typing import Optional

from pydantic import Field

from mcp_agent.agent.base import BaseAgent
from mcp_agent.agent.schema import AgentState, Memory


class ReActAgent(BaseAgent, ABC):
    """
    Implementation of the ReAct (Reasoning and Acting) agent pattern.
    
    This agent alternates between thinking (reasoning) and acting phases,
    allowing for more deliberate decision-making.
    """
    name: str
    description: Optional[str] = None

    system_prompt: Optional[str] = None
    next_step_prompt: Optional[str] = None

    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE

    max_steps: int = 10
    current_step: int = 0

    @abstractmethod
    async def think(self) -> bool:
        """Process current state and decide next action.
        
        Returns:
            bool: True if an action should be taken, False otherwise.
        """

    @abstractmethod
    async def act(self) -> str:
        """Execute decided actions.
        
        Returns:
            str: Result of the action.
        """

    async def step(self) -> str:
        """Execute a single step: think and act.
        
        Returns:
            str: Result of the step.
        """
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        return await self.act()
