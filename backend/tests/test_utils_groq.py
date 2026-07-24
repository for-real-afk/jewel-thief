from types import SimpleNamespace

import pytest

import utils


def _fake_response(json_body, status_code=200):
    resp = SimpleNamespace(status_code=status_code, ok=status_code < 400)
    resp.json = lambda: json_body
    resp.text = ""
    return resp


def test_groq_vision_chat_sends_bearer_auth_and_base64_image(mocker, valid_jpeg_bytes):
    mocker.patch.object(utils.settings, "groq_api_key", "gsk_test_key")
    mock_post = mocker.patch.object(
        utils.requests,
        "post",
        return_value=_fake_response({"choices": [{"message": {"content": "a gold necklace"}}]}),
    )

    result = utils.groq_vision_chat(valid_jpeg_bytes, "Describe this.", "qwen/qwen3.6-27b")

    assert result == "a gold necklace"
    url = mock_post.call_args.args[0]
    kwargs = mock_post.call_args.kwargs
    assert url == f"{utils.settings.groq_base_url}/chat/completions"
    assert kwargs["headers"] == {"Authorization": "Bearer gsk_test_key"}
    body = kwargs["json"]
    assert body["model"] == "qwen/qwen3.6-27b"
    content_parts = body["messages"][0]["content"]
    assert content_parts[0] == {"type": "text", "text": "Describe this."}
    assert content_parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_groq_vision_chat_downscales_image_before_sending(mocker, image_bytes_factory):
    large_image = image_bytes_factory(size=(1024, 1024))
    mock_post = mocker.patch.object(
        utils.requests,
        "post",
        return_value=_fake_response({"choices": [{"message": {"content": "ok"}}]}),
    )

    utils.groq_vision_chat(large_image, "Describe this.", "qwen/qwen3.6-27b")

    sent_b64 = mock_post.call_args.kwargs["json"]["messages"][0]["content"][1]["image_url"]["url"]
    # base64 payload should be meaningfully smaller than the original 1024px image
    assert len(sent_b64) < len(large_image)


def test_groq_vision_chat_raises_with_response_body_on_http_error(mocker, valid_jpeg_bytes, no_sleep):
    mocker.patch.object(
        utils.requests,
        "post",
        return_value=SimpleNamespace(status_code=413, ok=False, text="payload too large: max 1MB"),
    )

    with pytest.raises(Exception, match="payload too large"):
        utils.groq_vision_chat(valid_jpeg_bytes, "prompt", "qwen/qwen3.6-27b")


def test_groq_vision_chat_retries_on_transient_failure(mocker, no_sleep, valid_jpeg_bytes):
    mock_post = mocker.patch.object(
        utils.requests,
        "post",
        side_effect=[
            Exception("connection reset"),
            _fake_response({"choices": [{"message": {"content": "ok"}}]}),
        ],
    )

    result = utils.groq_vision_chat(valid_jpeg_bytes, "prompt", "qwen/qwen3.6-27b")

    assert result == "ok"
    assert mock_post.call_count == 2


def test_groq_vision_chat_reraises_after_persistent_failure(mocker, no_sleep, valid_jpeg_bytes):
    mock_post = mocker.patch.object(utils.requests, "post", side_effect=Exception("server down"))

    with pytest.raises(Exception, match="server down"):
        utils.groq_vision_chat(valid_jpeg_bytes, "prompt", "qwen/qwen3.6-27b")

    assert mock_post.call_count == 3
