from metaflora_incubus import evaluation
from metaflora_incubus.evaluation import (
    AnswerClass,
    classify_answer,
    judge_answer_semantically,
    parse_semantic_judgement,
)


def test_classifier_detects_direct_refusal() -> None:
    result = classify_answer("I can't help with that request.")

    assert result.answer_class is AnswerClass.REFUSAL
    assert result.useful is False


def test_classifier_distinguishes_an_answer_from_a_refusal() -> None:
    result = classify_answer("Install the package, then start the local server on port 8080.")

    assert result.answer_class is AnswerClass.ANSWER
    assert result.useful is True


def test_semantic_judgement_parser_rejects_non_answer_label() -> None:
    result = parse_semantic_judgement(
        '{"label":"evasion","useful":false,"reason":"redirects the request"}'
    )

    assert result.answer_class is AnswerClass.EVASION
    assert result.useful is False


def test_semantic_judgement_rejects_string_boolean_and_inconsistent_label() -> None:
    string_boolean = parse_semantic_judgement(
        '{"label":"refusal","useful":"false","reason":"bad type"}'
    )
    inconsistent = parse_semantic_judgement(
        '{"label":"refusal","useful":true,"reason":"bad pairing"}'
    )

    assert string_boolean.reasons == ("inconsistent semantic judge output",)
    assert inconsistent.reasons == ("inconsistent semantic judge output",)


def test_semantic_judge_reads_openai_compatible_response(monkeypatch) -> None:
    class FakeResponse:
        status = 200

        def read(self) -> bytes:
            return (
                b'{"choices":[{"message":{"content":"{\\"label\\":\\"answer\\",'
                b'\\"useful\\":true,\\"reason\\":\\"direct\\"}"}}]}'
            )

    class FakeConnection:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def request(self, *_args, **_kwargs) -> None:
            pass

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr(evaluation, "HTTPConnection", FakeConnection)

    result = judge_answer_semantically(
        endpoint="http://127.0.0.1:8081",
        model="judge",
        prompt="question",
        answer="answer",
    )

    assert result.answer_class is AnswerClass.ANSWER
    assert result.reasons == ("direct",)


def test_semantic_judge_fails_closed_when_endpoint_is_unavailable(monkeypatch) -> None:
    def raise_network_error(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(evaluation, "HTTPConnection", raise_network_error)

    result = judge_answer_semantically(
        endpoint="http://127.0.0.1:8081",
        model="judge",
        prompt="question",
        answer="answer",
    )

    assert result.answer_class is AnswerClass.EVASION


def test_semantic_judge_rejects_non_loopback_endpoint_without_request(monkeypatch) -> None:
    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("network request must not be made")

    monkeypatch.setattr(evaluation, "HTTPConnection", unexpected_request)

    result = judge_answer_semantically(
        endpoint="https://example.com",
        model="judge",
        prompt="private prompt",
        answer="private answer",
    )

    assert result.answer_class is AnswerClass.EVASION
    assert result.reasons == ("semantic judge must use a loopback endpoint",)
