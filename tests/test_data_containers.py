from smif import StateData


class TestStateData():
    """Test StateData object
    """
    def test_create(self):
        """StateData creation and accessors
        """
        val = StateData(1, {'test': "data"})

        assert val.target == 1
        assert val.data == {'test': "data"}
