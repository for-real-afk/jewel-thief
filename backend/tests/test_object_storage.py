import object_storage


def test_upload_catalog_image_puts_object_and_returns_public_url(mocker):
    mock_put = mocker.patch.object(object_storage._client, "put_object")
    mocker.patch.object(object_storage.settings, "r2_bucket_name", "test-bucket")
    mocker.patch.object(object_storage.settings, "r2_public_url_base", "https://pub-test.r2.dev")

    url = object_storage.upload_catalog_image("ring-1", b"fake jpeg bytes")

    mock_put.assert_called_once_with(
        Bucket="test-bucket", Key="catalog/ring-1.jpg", Body=b"fake jpeg bytes", ContentType="image/jpeg"
    )
    assert url == "https://pub-test.r2.dev/catalog/ring-1.jpg"


def test_delete_catalog_image_calls_delete_object(mocker):
    mock_delete = mocker.patch.object(object_storage._client, "delete_object")
    mocker.patch.object(object_storage.settings, "r2_bucket_name", "test-bucket")

    object_storage.delete_catalog_image("ring-1")

    mock_delete.assert_called_once_with(Bucket="test-bucket", Key="catalog/ring-1.jpg")


def test_key_for_uses_catalog_prefix_and_item_id():
    assert object_storage._key_for("abc-123") == "catalog/abc-123.jpg"
