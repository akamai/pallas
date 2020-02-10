import os


def test_athena(athena):
    info = athena.execute("SELECT 1")
    assert info["Status"]["State"] == "SUCCEEDED"
    assert (
        info["QueryExecutionContext"]["Database"] == os.environ["TEST_PALLAS_DATABASE"]
    )
