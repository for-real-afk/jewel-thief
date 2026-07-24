from types import SimpleNamespace

import pytest

import utils


def _fake_response(json_body, status_code=200):
    resp = SimpleNamespace(status_code=status_code)
    resp.json = lambda: json_body
    resp.raise_for_status = lambda: None
    return resp


def test_groq_vision_chat_sends_bearer_auth_and_base64_image(mocker):
    mocker.patch.object(utils.settings, "groq_api_key", "gsk_test_key")
    mock_post = mocker.patch.object(
        utils.requests,
        "post",
        return_value=_fake_response({"choices": [{"message": {"content": "a gold necklace"}}]}),
    )

    result = utils.groq_vision_chat(b"fake-jpeg-bytes", "Describe this.", "qwen/qwen3.6-27b")

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


def test_groq_vision_chat_retries_on_transient_failure(mocker, no_sleep):
    mock_post = mocker.patch.object(
        utils.requests,
        "post",
        side_effect=[
            Exception("connection reset"),
            _fake_response({"choices": [{"message": {"content": "ok"}}]}),
        ],
    )

    result = utils.groq_vision_chat(b"bytes", "prompt", "qwen/qwen3.6-27b")

    assert result == "ok"
    assert mock_post.call_count == 2


def test_groq_vision_chat_reraises_after_persistent_failure(mocker, no_sleep):
    mock_post = mocker.patch.object(utils.requests, "post", side_effect=Exception("server down"))

    with pytest.raises(Exception, match="server down"):
        utils.groq_vision_chat(b"bytes", "prompt", "qwen/qwen3.6-27b")

    assert mock_post.call_count == 3
