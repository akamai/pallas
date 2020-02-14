from pallas.results import QueryResults

sample_column_names = "id", "name"
sample_column_types = "integer", "varchar"

sample_data = [
    (1, "foo"),
    (2, "bar"),
]


sample_results = QueryResults(sample_column_names, sample_column_types, sample_data)


class TestQueryResults:
    def test_repr(self):
        assert repr(sample_results) == (
            "<QueryResults:"
            " 2 results,"
            " column_names=('id', 'name'),"
            " column_types=('integer', 'varchar')"
            ">"
        )

    def test_list(self):
        assert list(sample_results) == [
            {"id": 1, "name": "foo"},
            {"id": 2, "name": "bar"},
        ]

    def test_len(self):
        assert len(sample_results) == 2

    def test_get_item(self):
        assert sample_results[0] == {"id": 1, "name": "foo"}

    def test_get_slice(self):
        assert list(sample_results[:1]) == [{"id": 1, "name": "foo"}]

    def test_get_slice_from_end(self):
        assert list(sample_results[-1:]) == [{"id": 2, "name": "bar"}]
