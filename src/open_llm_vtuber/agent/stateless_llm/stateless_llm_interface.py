import abc
from typing import AsyncIterator, List, Dict, Any


class LLMProviderErrorText(str):
    """Provider failure text that is not assistant content.

    Adapters historically yielded a plain error string when the provider
    failed. Agents could not distinguish that string from a normal text
    chunk, so failures were persisted as chat history and counted as
    successful proactive deliveries. This ``str`` subclass keeps backwards
    compatibility for string-only consumers while giving delivery-sensitive
    callers a typed signal.
    """

    def __new__(cls, text: str, category: str = "provider_error"):
        instance = super().__new__(cls, text)
        instance.category = category
        return instance


class StatelessLLMInterface(metaclass=abc.ABCMeta):
    """
    Interface for a stateless language model.

    The word "stateless" means that the language model does not store memory,
    system prompts, or user messages, which is most of the LLM. If we send a
    message to the LLM, its response will be based on the message parameter alone.

    The StatelessLLMInterface class provides a method for generating chat
    completions asynchronously.

    We use StatelessLLMs to initialize Agents, which pack the StatelessLLM with
    memory, system prompts, and other features.

    """

    @abc.abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """
        Generates a chat completion asynchronously and return an iterator to the response.
        This function does not store memory or user messages.

        Parameters:
        - messages (List[Dict[str, Any]]): The list of messages to send to the API.
        - system (str, optional): System prompt to use for this completion.
        - tools (List[Dict[str, str]], optional): List of tools to use for this completion.
            - Each tool should follow the format:
            {
                "name": "tool_name",
                "description": "tool_description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {
                            "type": "string",
                            "description": "Description of param1"
                        },
                        "param2": {
                            "type": "integer",
                            "description": "Description of param2"
                        }
                    },
                    "required": ["param1"]
                }
            }

        Yields:
        - str: The content of each chunk from the API response.

        Raises:
        - APIConnectionError: When the server cannot be reached
        - RateLimitError: When a 429 status code is received
        - APIError: For other API-related errors
        """
        raise NotImplementedError
