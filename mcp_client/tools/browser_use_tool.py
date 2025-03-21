import logging
from typing import Any, Optional
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from pydantic import Field
from .base import BaseTool, ToolResult

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

    # Define Pydantic fields for browser components
    playwright: Any = Field(default=None)
    browser: Optional[Browser] = Field(default=None)
    context: Optional[BrowserContext] = Field(default=None)
    page: Optional[Page] = Field(default=None)

    async def setup(self):
        """Set up the browser if not already initialized."""
        logger.debug("Setting up browser...")
        if not self.browser:
            try:
                logger.debug("Starting Playwright...")
                self.playwright = await async_playwright().start()
                logger.debug("Launching Chromium browser...")
                self.browser = await self.playwright.chromium.launch(
                    headless=False,
                    args=['--start-maximized']
                )
                logger.debug("Creating new browser context...")
                self.context = await self.browser.new_context(
                    viewport={'width': 1920, 'height': 1080}
                )
                logger.debug("Creating new page...")
                self.page = await self.context.new_page()
                logger.debug("Browser setup complete")
            except Exception as e:
                logger.error(f"Error setting up browser: {str(e)}")
                if self.playwright:
                    await self.playwright.stop()
                raise

    async def cleanup(self):
        """Clean up browser resources."""
        logger.debug("Cleaning up browser resources...")
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        finally:
            self.page = None
            self.context = None
            self.browser = None
            self.playwright = None
            logger.debug("Browser cleanup complete")

    async def execute(self, **kwargs) -> Any:
        """Execute a browser action."""
        logger.debug(f"Executing browser action with args: {kwargs}")
        try:
            await self.setup()

            action = kwargs.get("action")
            if not action:
                return ToolResult(error="Action is required")

            if action == "navigate":
                url = kwargs.get("url")
                if not url:
                    return ToolResult(error="URL is required for navigate action")
                logger.debug(f"Navigating to URL: {url}")
                await self.page.goto(url)
                logger.debug("Navigation complete")
                return ToolResult(output=f"Navigated to {url}")

            elif action == "click":
                selector = kwargs.get("selector")
                if not selector:
                    return ToolResult(error="Selector is required for click action")
                logger.debug(f"Clicking element: {selector}")
                await self.page.click(selector)
                return ToolResult(output=f"Clicked element: {selector}")

            elif action == "type":
                selector = kwargs.get("selector")
                text = kwargs.get("text")
                if not selector or text is None:
                    return ToolResult(error="Selector and text are required for type action")
                logger.debug(f"Typing text into {selector}")
                await self.page.fill(selector, text)
                return ToolResult(output=f"Typed text into {selector}")

            elif action == "get_text":
                selector = kwargs.get("selector")
                if not selector:
                    return ToolResult(error="Selector is required for get_text action")
                logger.debug(f"Getting text from {selector}")
                text = await self.page.text_content(selector)
                return ToolResult(output=text)

            elif action == "get_links":
                logger.debug("Getting all links from page")
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
            logger.error(f"Error executing browser action: {str(e)}")
            return ToolResult(error=str(e))
        finally:
            # Don't cleanup here - let the cleanup method be called explicitly
            pass 