import os


class TestAthena:
    def test_execute(self, athena):
        info = athena.execute("SELECT 1")
        assert info["Status"]["State"] == "SUCCEEDED"
        assert (
            info["QueryExecutionContext"]["Database"]
            == os.environ["TEST_PALLAS_DATABASE"]
        )

    def test_submit(self, athena):
        query = athena.submit("SELECT 1")
        info = query.get_status()
        assert info["Status"]["State"] == "RUNNING"

    def test_kill(self, athena):
        query = athena.submit("SELECT 1")
        query.kill()
        info = query.get_status()
        assert info["Status"]["State"] == "CANCELLED"
