import os


class TestAthena:
    def test_execute(self, athena):
        info = athena.execute("SELECT 1")
        assert info.done
        assert info.succeeded
        assert info.state == "SUCCEEDED"
        assert info.sql == "SELECT 1"
        assert info.database == os.environ["TEST_PALLAS_DATABASE"]

    def test_submit(self, athena):
        query = athena.submit("SELECT 1")
        info = query.get_info()
        assert info.state == "RUNNING"
        assert not info.done
        assert not info.succeeded
        assert info.sql == "SELECT 1"
        assert info.database == os.environ["TEST_PALLAS_DATABASE"]

    def test_kill(self, athena):
        query = athena.submit("SELECT 1")
        query.kill()
        info = query.get_info()
        assert info.done
        assert not info.succeeded
        assert info.state == "CANCELLED"
        assert info.sql == "SELECT 1"
        assert info.database == os.environ["TEST_PALLAS_DATABASE"]
