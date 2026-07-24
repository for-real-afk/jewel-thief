import json
from types import SimpleNamespace

import reranker

IMAGE_QUERY = {"type": "image", "bytes": b"query-bytes"}


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

    result = reranker.rerank(IMAGE_QUERY, candidates)

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

    result = reranker.rerank(IMAGE_QUERY, candidates)

    assert len(result) == 2
    assert all(c["confidence"] == "medium" for c in result)
    assert all(c["reason"] == "no reranker judgment available" for c in result)


def test_response_missing_candidate_id_gets_fallback_reason(mocker):
    candidates = [_candidate("a", 0.9), _candidate("b", 0.7)]
    judgments = [{"id": "a", "confidence": "high", "reason": "matching cut"}]
    mocker.patch.object(
        reranker._client.models, "generate_content", return_value=_fake_llm_response(json.dumps(judgments))
    )

    result = reranker.rerank(IMAGE_QUERY, candidates)
    by_id = {c["id"]: c for c in result}

    assert by_id["a"]["confidence"] == "high"
    assert by_id["b"]["confidence"] == "medium"
    assert by_id["b"]["reason"] == "no reranker judgment available"


def test_empty_candidates_returns_empty_list_without_calling_llm(mocker):
    mock_generate = mocker.patch.object(reranker._client.models, "generate_content")

    result = reranker.rerank(IMAGE_QUERY, [])

    assert result == []
    mock_generate.assert_not_called()


def test_response_wrapped_in_think_block_is_still_parsed(mocker):
    candidates = [_candidate("a", 0.9)]
    judgments = [{"id": "a", "confidence": "high", "reason": "matching cut"}]
    wrapped = f"<think>\nlots of reasoning here about the image...\n</think>\n{json.dumps(judgments)}"
    mocker.patch.object(
        reranker._client.models, "generate_content", return_value=_fake_llm_response(wrapped)
    )

    result = reranker.rerank(IMAGE_QUERY, candidates)

    assert result[0]["confidence"] == "high"
    assert result[0]["reason"] == "matching cut"


def test_groq_provider_used_when_configured(mocker):
    mocker.patch.object(reranker.settings, "llm_provider", "groq")
    candidates = [_candidate("a", 0.9)]
    judgments = [{"id": "a", "confidence": "high", "reason": "matching cut"}]
    mock_groq = mocker.patch.object(reranker.utils, "groq_vision_chat", return_value=json.dumps(judgments))
    mock_gemini = mocker.patch.object(reranker._client.models, "generate_content")

    result = reranker.rerank(IMAGE_QUERY, candidates)

    assert result[0]["confidence"] == "high"
    mock_groq.assert_called_once()
    assert mock_groq.call_args.args[2] == reranker.settings.groq_model
    mock_gemini.assert_not_called()


# --- Image-query regression check: confirm the image Part is still built and
# sent exactly as before now that rerank() takes a query dict instead of raw
# bytes — this is the part most likely to break silently from the refactor. ---
def test_image_query_sends_image_part_to_gemini(mocker):
    candidates = [_candidate("a", 0.9)]
    mock_generate = mocker.patch.object(
        reranker._client.models, "generate_content",
        return_value=_fake_llm_response(json.dumps([{"id": "a", "confidence": "high", "reason": "x"}])),
    )

    reranker.rerank({"type": "image", "bytes": b"real-image-bytes"}, candidates)

    contents = mock_generate.call_args.kwargs["contents"]
    assert len(contents) == 2
    image_part = contents[0]
    assert image_part.inline_data.data == b"real-image-bytes"
    assert image_part.inline_data.mime_type == "image/jpeg"
    assert isinstance(contents[1], str)  # the prompt


# --- Text-query path ---
def test_text_query_sends_no_image_part_to_gemini(mocker):
    candidates = [_candidate("a", 0.9)]
    mock_generate = mocker.patch.object(
        reranker._client.models, "generate_content",
        return_value=_fake_llm_response(json.dumps([{"id": "a", "confidence": "high", "reason": "x"}])),
    )

    reranker.rerank({"type": "text", "text": "gold pink enamel earrings"}, candidates)

    contents = mock_generate.call_args.kwargs["contents"]
    assert len(contents) == 1  # no image Part, only the text prompt
    assert isinstance(contents[0], str)
    assert "gold pink enamel earrings" in contents[0]


def test_text_query_prompt_includes_query_string_and_richer_metadata(mocker):
    candidates = [{
        "id": "a", "score": 0.9,
        "metadata": {"name": "Studs", "category": "earrings", "caption": "gold with pink enamel detailing"},
    }]
    mock_generate = mocker.patch.object(
        reranker._client.models, "generate_content",
        return_value=_fake_llm_response(json.dumps([{"id": "a", "confidence": "high", "reason": "x"}])),
    )

    reranker.rerank({"type": "text", "text": "gold pink enamel earrings"}, candidates)

    prompt = mock_generate.call_args.kwargs["contents"][0]
    assert 'The customer searched for: "gold pink enamel earrings"' in prompt
    assert "caption=gold with pink enamel detailing" in prompt  # only sent for text queries


def test_image_query_prompt_omits_caption_description_tags(mocker):
    candidates = [{
        "id": "a", "score": 0.9,
        "metadata": {"name": "Studs", "category": "earrings", "caption": "should not appear"},
    }]
    mock_generate = mocker.patch.object(
        reranker._client.models, "generate_content",
        return_value=_fake_llm_response(json.dumps([{"id": "a", "confidence": "high", "reason": "x"}])),
    )

    reranker.rerank({"type": "image", "bytes": b"bytes"}, candidates)

    prompt = mock_generate.call_args.kwargs["contents"][1]
    assert "should not appear" not in prompt


def test_text_query_empty_candidates_returns_empty_without_calling_llm(mocker):
    mock_generate = mocker.patch.object(reranker._client.models, "generate_content")

    result = reranker.rerank({"type": "text", "text": "gold ring"}, [])

    assert result == []
    mock_generate.assert_not_called()


def test_text_query_uses_groq_text_chat_not_vision_chat(mocker):
    mocker.patch.object(reranker.settings, "llm_provider", "groq")
    candidates = [_candidate("a", 0.9)]
    judgments = [{"id": "a", "confidence": "high", "reason": "material match"}]
    mock_text = mocker.patch.object(reranker.utils, "groq_text_chat", return_value=json.dumps(judgments))
    mock_vision = mocker.patch.object(reranker.utils, "groq_vision_chat")

    result = reranker.rerank({"type": "text", "text": "gold pink enamel earrings"}, candidates)

    assert result[0]["confidence"] == "high"
    mock_text.assert_called_once()
    assert mock_text.call_args.args[1] == reranker.settings.groq_model
    mock_vision.assert_not_called()
