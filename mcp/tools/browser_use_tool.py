from typing import Any, Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from .base import BaseTool, ToolResult


class BrowserUseTool(BaseTool):
    """A tool for browser automation."""

    name: str = "browser_use"
    description: str = """Browser automation tool for web interactions.
Provides capabilities for navigating websites, clicking elements, typing text, and extracting content."""
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "navigate",
                    "click",
                    "type",
                    "get_text",
                    "get_links"
                ],
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'navigate' action",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for elements to interact with",
            },
            "text": {
                "type": "string",
                "description": "Text for 'type' action",
            },
        },
        "required": ["action"],
        "dependencies": {
            "navigate": ["url"],
            "click": ["selector"],
            "type": ["selector", "text"],
            "get_text": ["selector"],
            "get_links": [],
        },
    }

    browser: Optional[Browser] = None
    context: Optional[BrowserContext] = None
    page: Optional[Page] = None

    async def setup(self):
        """Set up the browser if not already initialized."""
        if not self.browser:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch()
            self.context = await self.browser.new_context()
            self.page = await self.context.new_page()

    async def cleanup(self):
        """Clean up browser resources."""
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.context = None
            self.page = None

    async def execute(self, **kwargs) -> Any:
        """Execute a browser action."""
        await self.setup()

        action = kwargs.get("action")
        if not action:
            return ToolResult(error="Action is required")

        try:
            if action == "navigate":
                url = kwargs.get("url")
                if not url:
                    return ToolResult(error="URL is required for navigate action")
                await self.page.goto(url)
                return ToolResult(output=f"Navigated to {url}")

            elif action == "click":
                selector = kwargs.get("selector")
                if not selector:
                    return ToolResult(error="Selector is required for click action")
                await self.page.click(selector)
                return ToolResult(output=f"Clicked element: {selector}")

            elif action == "type":
                selector = kwargs.get("selector")
                text = kwargs.get("text")
                if not selector or text is None:
                    return ToolResult(error="Selector and text are required for type action")
                await self.page.fill(selector, text)
                return ToolResult(output=f"Typed text into {selector}")

            elif action == "get_text":
                selector = kwargs.get("selector")
                if not selector:
                    return ToolResult(error="Selector is required for get_text action")
                text = await self.page.text_content(selector)
                return ToolResult(output=text)

            elif action == "get_links":
                links = await self.page.eval_on_selector_all("a[href]", """
                    elements => elements.map(el => ({
                        text: el.textContent,
                        href: el.href
                    }))
                """)
                return ToolResult(output=links)

            else:
                return ToolResult(error=f"Unknown action: {action}")

        except Exception as e:
            return ToolResult(error=str(e)) 