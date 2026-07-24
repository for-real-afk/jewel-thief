import json
from types import SimpleNamespace

import reranker


def _fake_llm_response(text):
    return SimpleNamespace(text=text)


def _candidate(id_, score, category="ring"):
    return {"id": id_, "score": score, "metadata": {"category": category}}


def test_wellformed_response_enriches_and_sorts_high_before_medium_low(mocker):
    candidates = [
        _candidate("a", 0.90),  # low
        _candidate("b", 0.70),  # high
        _candidate("c", 0.85),  # medium
        _candidate("d", 0.95),  # high
    ]
    judgments = [
        {"id": "a", "confidence": "low", "reason": "different gemstone cut"},
        {"id": "b", "confidence": "high", "reason": "matching metal finish"},
        {"id": "c", "confidence": "medium", "reason": "similar silhouette"},
        {"id": "d", "confidence": "high", "reason": "matching band pattern"},
    ]
    mocker.patch.object(
        reranker._client.models, "generate_content", return_value=_fake_llm_response(json.dumps(judgments))
    )

    result = reranker.rerank(b"query-bytes", candidates)

    assert [c["id"] for c in result] == ["d", "b", "c", "a"]
    for c in result:
        assert "confidence" in c and "reason" in c


def test_malformed_json_falls_back_to_medium_confidence_without_raising(mocker):
    candidates = [_candidate("a", 0.9), _candidate("b", 0.7)]
    # Truncated / markdown-fenced, not valid JSON on its own.
    mocker.patch.object(
        reranker._client.models,
        "generate_content",
        return_value=_fake_llm_response('```json\n[{"id": "a", "confidence": "high"'),
    )

    result = reranker.rerank(b"query-bytes", candidates)

    assert len(result) == 2
    assert all(c["confidence"] == "medium" for c in result)
    assert all(c["reason"] == "no reranker judgment available" for c in result)


def test_response_missing_candidate_id_gets_fallback_reason(mocker):
    candidates = [_candidate("a", 0.9), _candidate("b", 0.7)]
    judgments = [{"id": "a", "confidence": "high", "reason": "matching cut"}]
    mocker.patch.object(
        reranker._client.models, "generate_content", return_value=_fake_llm_response(json.dumps(judgments))
    )

    result = reranker.rerank(b"query-bytes", candidates)
    by_id = {c["id"]: c for c in result}

    assert by_id["a"]["confidence"] == "high"
    assert by_id["b"]["confidence"] == "medium"
    assert by_id["b"]["reason"] == "no reranker judgment available"


def test_empty_candidates_returns_empty_list_without_calling_llm(mocker):
    mock_generate = mocker.patch.object(reranker._client.models, "generate_content")

    result = reranker.rerank(b"query-bytes", [])

    assert result == []
    mock_generate.assert_not_called()


def test_response_wrapped_in_think_block_is_still_parsed(mocker):
    candidates = [_candidate("a", 0.9)]
    judgments = [{"id": "a", "confidence": "high", "reason": "matching cut"}]
    wrapped = f"<think>\nlots of reasoning here about the image...\n</think>\n{json.dumps(judgments)}"
    mocker.patch.object(
        reranker._client.models, "generate_content", return_value=_fake_llm_response(wrapped)
    )

    result = reranker.rerank(b"query-bytes", candidates)

    assert result[0]["confidence"] == "high"
    assert result[0]["reason"] == "matching cut"


def test_groq_provider_used_when_configured(mocker):
    mocker.patch.object(reranker.settings, "llm_provider", "groq")
    candidates = [_candidate("a", 0.9)]
    judgments = [{"id": "a", "confidence": "high", "reason": "matching cut"}]
    mock_groq = mocker.patch.object(reranker.utils, "groq_vision_chat", return_value=json.dumps(judgments))
    mock_gemini = mocker.patch.object(reranker._client.models, "generate_content")

    result = reranker.rerank(b"query-bytes", candidates)

    assert result[0]["confidence"] == "high"
    mock_groq.assert_called_once()
    assert mock_groq.call_args.args[2] == reranker.settings.groq_model
    mock_gemini.assert_not_called()
