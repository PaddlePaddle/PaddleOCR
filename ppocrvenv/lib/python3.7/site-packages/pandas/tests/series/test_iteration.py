class TestIteration:
    def test_keys(self, datetime_series):
        assert datetime_series.keys() is datetime_series.index

    def test_iter_datetimes(self, datetime_series):
        for i, val in enumerate(datetime_series):
            assert val == datetime_series[i]

    def test_iter_strings(self, string_series):
        for i, val in enumerate(string_series):
            assert val == string_series[i]

    def test_iteritems_datetimes(self, datetime_series):
        for idx, val in datetime_series.iteritems():
            assert val == datetime_series[idx]

    def test_iteritems_strings(self, string_series):
        for idx, val in string_series.iteritems():
            assert val == string_series[idx]

        # assert is lazy (generators don't define reverse, lists do)
        assert not hasattr(string_series.iteritems(), "reverse")

    def test_items_datetimes(self, datetime_series):
        for idx, val in datetime_series.items():
            assert val == datetime_series[idx]

    def test_items_strings(self, string_series):
        for idx, val in string_series.items():
            assert val == string_series[idx]

        # assert is lazy (generators don't define reverse, lists do)
        assert not hasattr(string_series.items(), "reverse")
