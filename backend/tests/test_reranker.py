import json
from types import SimpleNamespace

import pytest

from app import reranker

IMAGE_QUERY = {"type": "image", "bytes": b"query-bytes"}


def _fake_llm_response(text):
    return SimpleNamespace(text=text)


def _candidate(id_, score, category="ring"):
    return {"id": id_, "score": score, "metadata": {"category": category}}


def test_wellformed_response_sorts_by_final_rank_score_not_confidence_tier_first(mocker):
    # Scores chosen to avoid exact ties: "b" has HIGH confidence but a much
    # lower cosine score than "a" and "c" -- under the old confidence-tier
    # -first sort it would have outranked both; under final_rank_score
    # (score-dominant, confidence a bounded +/-0.05 nudge) it must not.
    candidates = [
        _candidate("a", 0.90),  # low    -> 0.90 - 0.05 = 0.85
        _candidate("b", 0.60),  # high   -> 0.60 + 0.05 = 0.65
        _candidate("c", 0.85),  # medium -> 0.85 + 0.00 = 0.85
        _candidate("d", 0.95),  # high   -> 0.95 + 0.05 = 1.00
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

    assert [c["id"] for c in result] == ["d", "a", "c", "b"]
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


# --- Text-query path (LLM rerank(), used only when the cheap path is ambiguous) ---
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


# --- final_rank_score ---
def test_final_rank_score_medium_with_higher_score_beats_high_with_lower_score():
    medium = {"score": 0.75, "confidence": "medium"}
    high = {"score": 0.55, "confidence": "high"}

    assert reranker.final_rank_score(medium) > reranker.final_rank_score(high)


def test_final_rank_score_high_still_breaks_a_near_tie():
    high = {"score": 0.60, "confidence": "high"}
    medium = {"score": 0.58, "confidence": "medium"}

    assert reranker.final_rank_score(high) > reranker.final_rank_score(medium)


# --- score_candidates_cheap (Step 5: default, zero-API-call text ranking) ---
def test_score_candidates_cheap_blends_similarity_and_lexical_overlap():
    candidates = [
        {"id": "a", "score": 0.9, "metadata": {"name": "Gold Ring", "caption": "a gold ring with a ruby"}},
        {"id": "b", "score": 0.9, "metadata": {"name": "Silver Necklace", "caption": "a plain silver chain"}},
    ]

    result = reranker.score_candidates_cheap("gold ring ruby", candidates)

    by_id = {c["id"]: c for c in result}
    # "a": all 3 query terms ("gold", "ring", "ruby") appear -> overlap=1.0 -> blended = 0.7*0.9 + 0.3*1.0 = 0.93
    assert by_id["a"]["blended_score"] == pytest.approx(0.93)
    # "b": none of the query terms appear -> overlap=0.0 -> blended = 0.7*0.9 + 0.3*0.0 = 0.63
    assert by_id["b"]["blended_score"] == pytest.approx(0.63)
    assert result[0]["id"] == "a"  # sorted best-first


def test_score_candidates_cheap_confidence_thresholds():
    candidates = [
        {"id": "hi", "score": 1.0, "metadata": {"caption": "match match match"}},  # overlap=1.0, blended=1.0
        {"id": "mid", "score": 0.6, "metadata": {"caption": "no overlap here"}},  # blended = 0.7*0.6 = 0.42
        {"id": "lo", "score": 0.1, "metadata": {"caption": "no overlap here"}},  # blended = 0.7*0.1 = 0.07
    ]

    result = reranker.score_candidates_cheap("match", candidates)
    by_id = {c["id"]: c for c in result}

    assert by_id["hi"]["confidence"] == "high"    # blended 1.0 > 0.75
    assert by_id["mid"]["confidence"] == "low"    # blended 0.42 <= 0.5
    assert by_id["lo"]["confidence"] == "low"     # blended 0.07 <= 0.5


def test_score_candidates_cheap_empty_candidates_returns_empty():
    assert reranker.score_candidates_cheap("gold ring", []) == []


def test_score_candidates_cheap_and_rerank_use_the_same_final_rank_score(mocker):
    # Regression guard against the two ranking paths drifting apart: if
    # final_rank_score changes, BOTH score_candidates_cheap() and rerank()
    # must reflect that change, since both are supposed to call the same
    # shared function rather than each doing their own sort.
    mocker.patch.object(reranker, "final_rank_score", lambda c: -c["score"])  # inverted: worst-first

    cheap_result = reranker.score_candidates_cheap(
        "x", [{"id": "a", "score": 0.9, "metadata": {}}, {"id": "b", "score": 0.1, "metadata": {}}]
    )
    assert [c["id"] for c in cheap_result] == ["b", "a"]  # worst-first, confirms the monkeypatch took effect

    mocker.patch.object(
        reranker._client.models, "generate_content",
        return_value=_fake_llm_response(json.dumps([
            {"id": "a", "confidence": "high", "reason": "x"}, {"id": "b", "confidence": "high", "reason": "y"},
        ])),
    )
    rerank_result = reranker.rerank(
        IMAGE_QUERY, [{"id": "a", "score": 0.9, "metadata": {}}, {"id": "b", "score": 0.1, "metadata": {}}]
    )
    assert [c["id"] for c in rerank_result] == ["b", "a"]  # same inverted order


# --- is_plausibly_jewelry (Step 12: domain gate) ---
def test_is_plausibly_jewelry_true_for_yes_response(mocker):
    mocker.patch.object(
        reranker._client.models, "generate_content", return_value=_fake_llm_response("yes")
    )

    assert reranker.is_plausibly_jewelry(b"fake-jpeg-bytes") is True


def test_is_plausibly_jewelry_false_for_no_response(mocker):
    mocker.patch.object(
        reranker._client.models, "generate_content", return_value=_fake_llm_response("no")
    )

    assert reranker.is_plausibly_jewelry(b"fake-jpeg-bytes") is False


def test_is_plausibly_jewelry_sends_image_and_constrained_output(mocker):
    mock_generate = mocker.patch.object(
        reranker._client.models, "generate_content", return_value=_fake_llm_response("yes")
    )

    reranker.is_plausibly_jewelry(b"fake-jpeg-bytes")

    kwargs = mock_generate.call_args.kwargs
    assert kwargs["contents"][0].inline_data.data == b"fake-jpeg-bytes"
    assert "yes or no" in kwargs["contents"][1].lower()
    assert kwargs["config"].max_output_tokens == 5
