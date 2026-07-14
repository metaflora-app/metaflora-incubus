import torch
from fastapi.testclient import TestClient

from metaflora_incubus.server import LocalRuntime, create_app


class FakeRuntime:
    def complete(self, messages, max_tokens: int, temperature: float) -> str:
        assert messages[0].content == "hello"
        assert max_tokens == 8
        assert temperature == 0
        return "world"


def test_server_exposes_only_its_local_model() -> None:
    client = TestClient(create_app(FakeRuntime()))

    assert client.get("/v1/models").json()["data"][0]["id"] == "metaflora-incubus-v1"
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "metaflora-incubus-v1",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "temperature": 0,
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "world"
    missing = client.post("/v1/chat/completions", json={"model": "wrong", "messages": []})
    assert missing.status_code == 404


def test_runtime_renders_chat_template_and_decodes_only_new_tokens() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            assert messages[0]["content"] == "hello"
            return "rendered"

        def __call__(self, _rendered, **_kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, tokens, **_kwargs):
            assert tokens.tolist() == [3, 4]
            return "completion"

    class Model:
        def __init__(self) -> None:
            self.parameter = torch.nn.Parameter(torch.ones(1))

        def parameters(self):
            return iter((self.parameter,))

        def generate(self, **_kwargs):
            return torch.tensor([[1, 2, 3, 4]])

    message = type(
        "Message", (), {"model_dump": lambda self: {"role": "user", "content": "hello"}}
    )()
    runtime = LocalRuntime(model_path=".", model=Model(), tokenizer=Tokenizer())
    content = runtime.complete([message], 8, 0)

    assert content == "completion"
