from types import SimpleNamespace

import pytest

import embeddings

FAKE_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def _fake_response(vector=None):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=vector or FAKE_VECTOR)])


def test_embed_image_sends_no_task_type(mocker):
    # task_type is NOT supported by gemini-embedding-2 and was silently
    # ignored -- confirmed against official docs and a filed llama_index bug.
    # embed_image must not pass it at all.
    mock_embed = mocker.patch.object(
        embeddings._client.models, "embed_content", return_value=_fake_response()
    )

    result = embeddings.embed_image(b"fake-jpeg-bytes")

    assert result == FAKE_VECTOR
    kwargs = mock_embed.call_args.kwargs
    assert not hasattr(kwargs["config"], "task_type") or kwargs["config"].task_type is None
    assert kwargs["contents"].inline_data.data == b"fake-jpeg-bytes"
    assert kwargs["config"].output_dimensionality == embeddings.settings.embedding_dimensions


def test_embed_text_query_applies_task_prefix(mocker):
    mock_embed = mocker.patch.object(
        embeddings._client.models, "embed_content", return_value=_fake_response()
    )

    result = embeddings.embed_text_query("gold pink enamel earrings")

    assert result == FAKE_VECTOR
    kwargs = mock_embed.call_args.kwargs
    assert kwargs["contents"] == "task: search result | query: gold pink enamel earrings"
    assert not hasattr(kwargs["config"], "task_type") or kwargs["config"].task_type is None
    assert kwargs["config"].output_dimensionality == embeddings.settings.embedding_dimensions


def test_embed_text_query_retries_on_transient_failure(mocker, no_sleep):
    mock_embed = mocker.patch.object(
        embeddings._client.models,
        "embed_content",
        side_effect=[Exception("transient network error"), _fake_response()],
    )

    result = embeddings.embed_text_query("gold ring")

    assert result == FAKE_VECTOR
    assert mock_embed.call_count == 2


def test_embed_catalog_item_sends_interleaved_image_and_text_in_one_call(mocker):
    mock_embed = mocker.patch.object(
        embeddings._client.models, "embed_content", return_value=_fake_response()
    )

    result = embeddings.embed_catalog_item(b"fake-jpeg-bytes", "name: Ruby Ring. A red ring. . Tags: gold, ruby.")

    assert result == FAKE_VECTOR
    assert mock_embed.call_count == 1  # one interleaved call, not two separate ones
    kwargs = mock_embed.call_args.kwargs
    contents = kwargs["contents"]
    assert isinstance(contents, list) and len(contents) == 2
    assert contents[0].inline_data.data == b"fake-jpeg-bytes"
    assert contents[1] == "name: Ruby Ring. A red ring. . Tags: gold, ruby."
    assert not hasattr(kwargs["config"], "task_type") or kwargs["config"].task_type is None
    assert kwargs["config"].output_dimensionality == embeddings.settings.embedding_dimensions


def test_embed_catalog_item_retries_on_transient_failure(mocker, no_sleep):
    mock_embed = mocker.patch.object(
        embeddings._client.models,
        "embed_content",
        side_effect=[Exception("transient network error"), _fake_response()],
    )

    result = embeddings.embed_catalog_item(b"fake-jpeg-bytes", "name: Ring.")

    assert result == FAKE_VECTOR
    assert mock_embed.call_count == 2


def test_persistent_failure_is_reraised_after_retry_attempts(mocker, no_sleep):
    mock_embed = mocker.patch.object(
        embeddings._client.models,
        "embed_content",
        side_effect=Exception("persistent network error"),
    )

    with pytest.raises(Exception, match="persistent network error"):
        embeddings.embed_image(b"fake-jpeg-bytes")

    assert mock_embed.call_count == 3


def test_caption_image_uses_gemini_by_default(mocker):
    mock_generate = mocker.patch.object(
        embeddings._client.models,
        "generate_content",
        return_value=SimpleNamespace(text="a gold ring with a diamond"),
    )

    result = embeddings.caption_image(b"fake-jpeg-bytes")

    assert result == "a gold ring with a diamond"
    mock_generate.assert_called_once()
