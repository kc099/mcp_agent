import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import aiohttp
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import config


class LLM:
    """A base class for language models."""

    @classmethod
    async def call(
        cls,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        n: int = 1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> str:
        """Call the language model."""
        raise NotImplementedError


class ChatOpenAI(LLM):
    """A wrapper around OpenAI's chat completion API."""

    @classmethod
    async def call(
        cls,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        n: int = 1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> str:
        """Call the OpenAI chat completion API."""
        async with AsyncOpenAI(
            api_key=config.openai_api_key,
            api_base=config.openai_api_base,
            api_type=config.openai_api_type,
            api_version=config.openai_api_version,
            organization=config.openai_organization,
        ) as openai:
            response = await openai.chat.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=n,
                stop=stop,
            )
            return response.choices[0].message.content


class ChatAnthropic(LLM):
    """A wrapper around Anthropic's chat completion API."""

    @classmethod
    async def call(
        cls,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        n: int = 1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> str:
        """Call the Anthropic chat completion API."""
        async with AsyncAnthropic(api_key=config.anthropic_api_key) as anthropic:
            response = await anthropic.completions.create(
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model="claude-v1",
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
            )
            return response.completions[0].text


class ChatHuggingFace(LLM):
    """A wrapper around HuggingFace's chat completion API."""

    @classmethod
    async def call(
        cls,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        n: int = 1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> str:
        """Call the HuggingFace chat completion API."""
        headers = {"Authorization": f"Bearer {config.huggingface_api_key}"}
        async with aiohttp.ClientSession(headers=headers) as session:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": 1.0,
                    "num_return_sequences": n,
                },
            }
            async with session.post(
                "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-1-pythia-12b",
                json=payload,
            ) as response:
                response_json = await response.json()
                return response_json[0]["generated_text"]


async def main():
    """Test the language models."""
    prompt = "What is the capital of France?"

    print("OpenAI:")
    print(await ChatOpenAI.call(prompt))

    print("Anthropic:")
    print(await ChatAnthropic.call(prompt))

    print("HuggingFace:")
    print(await ChatHuggingFace.call(prompt))


if __name__ == "__main__":
    asyncio.run(main()) 