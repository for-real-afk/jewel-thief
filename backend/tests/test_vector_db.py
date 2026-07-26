import vector_db


def _fake_item(i):
    return {"id": f"item-{i}", "vector": [0.1] * 8, "metadata": {"category": "ring"}}


def test_get_or_create_index_creates_when_missing(mocker):
    mocker.patch.object(vector_db._pc, "list_indexes", return_value=[{"name": "some-other-index"}])
    mock_create = mocker.patch.object(vector_db._pc, "create_index")
    mock_index_obj = mocker.Mock()
    mock_index_ctor = mocker.patch.object(vector_db._pc, "Index", return_value=mock_index_obj)

    result = vector_db.get_or_create_index()

    mock_create.assert_called_once()
    _, kwargs = mock_create.call_args
    assert kwargs["name"] == vector_db.settings.pinecone_index_name
    assert kwargs["dimension"] == vector_db.settings.embedding_dimensions
    assert kwargs["metric"] == "cosine"
    mock_index_ctor.assert_called_once_with(vector_db.settings.pinecone_index_name)
    assert result is mock_index_obj


def test_get_or_create_index_skips_when_already_exists(mocker):
    mocker.patch.object(
        vector_db._pc, "list_indexes", return_value=[{"name": vector_db.settings.pinecone_index_name}]
    )
    mock_create = mocker.patch.object(vector_db._pc, "create_index")
    mocker.patch.object(vector_db._pc, "Index", return_value=mocker.Mock())

    vector_db.get_or_create_index()

    mock_create.assert_not_called()


def test_upsert_batch_chunks_into_groups_of_100(mocker):
    mock_index = mocker.Mock()
    mocker.patch.object(vector_db, "get_or_create_index", return_value=mock_index)

    items = [_fake_item(i) for i in range(250)]
    count = vector_db.upsert_batch(items)

    assert count == 250
    assert mock_index.upsert.call_count == 3
    chunk_sizes = [len(call.kwargs["vectors"]) for call in mock_index.upsert.call_args_list]
    assert chunk_sizes == [100, 100, 50]


def test_search_passes_filter_through_unmodified_and_reshapes_results(mocker):
    mock_index = mocker.Mock()
    mock_index.query.return_value = {
        "matches": [
            {"id": "a", "score": 0.91, "metadata": {"category": "ring"}},
            {"id": "b", "score": 0.80, "metadata": {"category": "necklace"}},
        ]
    }
    mocker.patch.object(vector_db, "get_or_create_index", return_value=mock_index)

    metadata_filter = {"category": {"$eq": "ring"}, "price": {"$gte": 100.0}}
    results = vector_db.search([0.1, 0.2], top_k=5, metadata_filter=metadata_filter)

    mock_index.query.assert_called_once_with(
        vector=[0.1, 0.2], top_k=5, include_metadata=True, filter=metadata_filter
    )
    assert results == [
        {"id": "a", "score": 0.91, "metadata": {"category": "ring"}},
        {"id": "b", "score": 0.80, "metadata": {"category": "necklace"}},
    ]


def test_search_without_filter_uses_empty_dict_not_none(mocker):
    mock_index = mocker.Mock()
    mock_index.query.return_value = {"matches": []}
    mocker.patch.object(vector_db, "get_or_create_index", return_value=mock_index)

    vector_db.search([0.1, 0.2])

    kwargs = mock_index.query.call_args.kwargs
    assert kwargs["filter"] == {}
    assert kwargs["filter"] is not None


def test_search_clamps_score_overshoot_to_1(mocker):
    # Pinecone's approximate cosine computation can overshoot slightly past
    # 1.0 due to floating-point error in the ANN index.
    mock_index = mocker.Mock()
    mock_index.query.return_value = {"matches": [{"id": "a", "score": 1.004, "metadata": {}}]}
    mocker.patch.object(vector_db, "get_or_create_index", return_value=mock_index)

    results = vector_db.search([0.1, 0.2])

    assert results[0]["score"] == 1.0


def test_search_clamps_negative_score_to_0(mocker):
    mock_index = mocker.Mock()
    mock_index.query.return_value = {"matches": [{"id": "a", "score": -0.001, "metadata": {}}]}
    mocker.patch.object(vector_db, "get_or_create_index", return_value=mock_index)

    results = vector_db.search([0.1, 0.2])

    assert results[0]["score"] == 0.0


def test_search_leaves_in_range_score_unchanged(mocker):
    mock_index = mocker.Mock()
    mock_index.query.return_value = {"matches": [{"id": "a", "score": 0.734, "metadata": {}}]}
    mocker.patch.object(vector_db, "get_or_create_index", return_value=mock_index)

    results = vector_db.search([0.1, 0.2])

    assert results[0]["score"] == 0.734
