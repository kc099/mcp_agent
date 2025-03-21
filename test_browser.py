import asyncio
from playwright.async_api import async_playwright

async def main():
    print("Starting Playwright test...")
    async with async_playwright() as p:
        print("Launching browser...")
        browser = await p.chromium.launch(headless=False)
        print("Creating context...")
        context = await browser.new_context()
        print("Creating page...")
        page = await context.new_page()
        print("Navigating to Google...")
        await page.goto('https://www.google.com')
        print("Waiting for 5 seconds...")
        await asyncio.sleep(5)
        print("Closing browser...")
        await browser.close()
        print("Test complete!")

if __name__ == '__main__':
    asyncio.run(main()) 