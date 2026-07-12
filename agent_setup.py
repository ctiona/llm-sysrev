"""
agent_setup.py: unified LLM agent for systematic review screening.

Set OPENAI_BASE_URL and OPENAI_API_KEY as environment variables before running.
Any OpenAI-compatible endpoint is supported.
"""
import importlib
import os
from openai import OpenAI

# Loaded at startup by each framework script via load_syscontext().
# Defaults to syscontext.py if load_syscontext() is never called.
_components: dict = {}


def load_syscontext(module_name: str = "syscontext") -> None:
    """Import system-prompt components from *module_name* (no .py suffix)."""
    global _components
    _components = importlib.import_module(module_name).syscontext_components

_base_url = os.environ.get("OPENAI_BASE_URL", "")
_api_key = os.environ.get("OPENAI_API_KEY", "")

if not _base_url or not _api_key:
    raise EnvironmentError(
        "OPENAI_BASE_URL and OPENAI_API_KEY must be set as environment variables."
    )

client = OpenAI(base_url=_base_url, api_key=_api_key)


def get_models() -> None:
    for m in client.models.list().to_dict()["data"]:
        print(f"Name: {m['name']}\nid: {m['id']}\n")


def llmpass(model: str, context: str, prompt: str, **kwargs) -> tuple[str, str]:
    """
    Send a single request to the model and return (thought, reply).

    Supports two response conventions:
    - Structured reasoning field (e.g. o-series, QwQ): reads message.reasoning
    - <think>…</think> inline tag (e.g. DeepSeek, Nemotron): splits message.content
    """
    messages = [
        {"role": "system", "content": "detailed thinking on"},
        {"role": "user", "content": context.strip() + "\n\n" + prompt.strip()},
    ]
    response = client.chat.completions.create(
        model=model, messages=messages, stream=False, **kwargs
    )
    msg = response.choices[0].message
    thought = getattr(msg, "reasoning", None) or ""
    content = msg.content or ""

    if not thought and "</think>" in content:
        thought, content = content.split("</think>", 1)
        thought = thought.strip().removeprefix("<think>")

    return thought.strip(), content.strip().replace("–", "-")


class Agent:
    """
    Stateless LLM agent whose system prompt is assembled from modular
    components defined in syscontext.syscontext_components.

    Parameters
    ----------
    model : str
        Model identifier as accepted by the endpoint (e.g. "gpt-4o", "nemo-base").
    background : str
        Key into syscontext_components["Background"] (e.g. "Basic").
    task : str
        Key into syscontext_components["Task"].
    criteria : str
        Key into syscontext_components["Criteria"].
    output_format : str
        Key into syscontext_components["Output_format"].
    **generation_kwargs
        Optional generation parameters forwarded to the API call
        (e.g. temperature=0.6, top_p=0.95).
    """

    def __init__(
        self,
        model: str,
        background: str,
        task: str,
        criteria: str,
        output_format: str,
        **generation_kwargs,
    ) -> None:
        self.model = model
        self.generation_kwargs = generation_kwargs
        self.agent_context = self._compose_context(background, task, criteria, output_format)

    def _compose_context(
        self, background: str, task: str, criteria: str, output_format: str
    ) -> str:
        parts = [
            _components["Background"][background],
            _components["Task"][task],
            _components["Criteria"][criteria],
            _components["Output_format"][output_format],
        ]
        return "\n".join(parts).strip()

    def reply_to(self, prompt: str) -> tuple[str, str]:
        return llmpass(self.model, self.agent_context, prompt, **self.generation_kwargs)
