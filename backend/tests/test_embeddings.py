from types import SimpleNamespace

import pytest

import embeddings

FAKE_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def _fake_response(vector=None):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=vector or FAKE_VECTOR)])


def test_embed_query_image_uses_retrieval_query_task_type(mocker):
    mock_embed = mocker.patch.object(
        embeddings._client.models, "embed_content", return_value=_fake_response()
    )

    result = embeddings.embed_query_image(b"fake-jpeg-bytes")

    assert result == FAKE_VECTOR
    kwargs = mock_embed.call_args.kwargs
    assert kwargs["config"].task_type == "RETRIEVAL_QUERY"


def test_embed_catalog_image_uses_retrieval_document_task_type(mocker):
    mock_embed = mocker.patch.object(
        embeddings._client.models, "embed_content", return_value=_fake_response()
    )

    result = embeddings.embed_catalog_image(b"fake-jpeg-bytes")

    assert result == FAKE_VECTOR
    kwargs = mock_embed.call_args.kwargs
    assert kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_output_dimensionality_matches_settings(mocker):
    mock_embed = mocker.patch.object(
        embeddings._client.models, "embed_content", return_value=_fake_response()
    )

    embeddings.embed_catalog_image(b"fake-jpeg-bytes")

    kwargs = mock_embed.call_args.kwargs
    assert kwargs["config"].output_dimensionality == embeddings.settings.embedding_dimensions


def test_embed_batch_calls_embed_once_per_image_in_order(mocker):
    mock_embed = mocker.patch.object(
        embeddings._client.models, "embed_content", return_value=_fake_response()
    )

    images = [b"img-1", b"img-2", b"img-3"]
    result = embeddings.embed_batch_catalog_images(images)

    assert mock_embed.call_count == 3
    assert result == [FAKE_VECTOR, FAKE_VECTOR, FAKE_VECTOR]
    called_bytes = [call.kwargs["contents"].inline_data.data for call in mock_embed.call_args_list]
    assert called_bytes == images


def test_transient_failure_then_success_retries_and_succeeds(mocker, no_sleep):
    mock_embed = mocker.patch.object(
        embeddings._client.models,
        "embed_content",
        side_effect=[Exception("transient network error"), _fake_response()],
    )

    result = embeddings.embed_query_image(b"fake-jpeg-bytes")

    assert result == FAKE_VECTOR
    assert mock_embed.call_count == 2


def test_persistent_failure_is_reraised_after_retry_attempts(mocker, no_sleep):
    mock_embed = mocker.patch.object(
        embeddings._client.models,
        "embed_content",
        side_effect=Exception("persistent network error"),
    )

    with pytest.raises(Exception, match="persistent network error"):
        embeddings.embed_query_image(b"fake-jpeg-bytes")

    assert mock_embed.call_count == 3


def test_embed_text_query_uses_retrieval_query_task_type(mocker):
    mock_embed = mocker.patch.object(
        embeddings._client.models, "embed_content", return_value=_fake_response()
    )

    result = embeddings.embed_text_query("gold pink enamel earrings")

    assert result == FAKE_VECTOR
    kwargs = mock_embed.call_args.kwargs
    assert kwargs["config"].task_type == "RETRIEVAL_QUERY"
    assert kwargs["contents"] == "gold pink enamel earrings"
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


def test_caption_image_uses_gemini_by_default(mocker):
    mock_generate = mocker.patch.object(
        embeddings._client.models,
        "generate_content",
        return_value=SimpleNamespace(text="a gold ring with a diamond"),
    )

    result = embeddings.caption_image(b"fake-jpeg-bytes")

    assert result == "a gold ring with a diamond"
    mock_generate.assert_called_once()


