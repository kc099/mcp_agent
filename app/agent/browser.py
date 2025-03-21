import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from pydantic import Field

from app.agent.base import Agent
from app.config import config
from app.llm import ChatOpenAI
from app.prompt.browser import NEXT_STEP_PROMPT
from app.schema import Step
from app.tool.base import ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.terminate import Terminate


class BrowserAgent(Agent):
    """
    An agent that uses a web browser to interact with web pages.

    This agent uses a browser tool to navigate and interact with web pages.
    It can be used to automate web tasks or to extract information from websites.
    """

    name: str = "BrowserAgent"
    description: str = "An agent that uses a web browser to interact with web pages"

    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(BrowserUseTool(), Terminate())
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.browser_tool = self.available_tools.get("BrowserUseTool")
        self.browser_tool.agent = self

    async def observe(self, url: str) -> str:
        """Observe the current page in the browser."""
        return await self.browser_tool.observe(url)

    async def _take_next_step(self) -> Optional[Step]:
        """Take the next step in the agent's plan."""
        # Get the latest observation
        observation = await self.observe(self.browser_tool.current_url)

        # Construct the prompt
        prompt = self.next_step_prompt.format(
            url=self.browser_tool.current_url,
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

    async def run(self) -> str:
        """Run the agent."""
        self.steps: List[Step] = []

        # Take steps until a terminal state is reached
        while True:
            step = await self._take_next_step()
            if step is None:
                break

            self.steps.append(step)

            if step.tool == "Terminate":
                break

        # Return the result
        return self.steps[-1].result 