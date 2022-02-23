import json

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Index,
    Series,
    json_normalize,
)
import pandas._testing as tm

from pandas.io.json._normalize import nested_to_record


@pytest.fixture
def deep_nested():
    # deeply nested data
    return [
        {
            "country": "USA",
            "states": [
                {
                    "name": "California",
                    "cities": [
                        {"name": "San Francisco", "pop": 12345},
                        {"name": "Los Angeles", "pop": 12346},
                    ],
                },
                {
                    "name": "Ohio",
                    "cities": [
                        {"name": "Columbus", "pop": 1234},
                        {"name": "Cleveland", "pop": 1236},
                    ],
                },
            ],
        },
        {
            "country": "Germany",
            "states": [
                {"name": "Bayern", "cities": [{"name": "Munich", "pop": 12347}]},
                {
                    "name": "Nordrhein-Westfalen",
                    "cities": [
                        {"name": "Duesseldorf", "pop": 1238},
                        {"name": "Koeln", "pop": 1239},
                    ],
                },
            ],
        },
    ]


@pytest.fixture
def state_data():
    return [
        {
            "counties": [
                {"name": "Dade", "population": 12345},
                {"name": "Broward", "population": 40000},
                {"name": "Palm Beach", "population": 60000},
            ],
            "info": {"governor": "Rick Scott"},
            "shortname": "FL",
            "state": "Florida",
        },
        {
            "counties": [
                {"name": "Summit", "population": 1234},
                {"name": "Cuyahoga", "population": 1337},
            ],
            "info": {"governor": "John Kasich"},
            "shortname": "OH",
            "state": "Ohio",
        },
    ]


@pytest.fixture
def author_missing_data():
    return [
        {"info": None},
        {
            "info": {"created_at": "11/08/1993", "last_updated": "26/05/2012"},
            "author_name": {"first": "Jane", "last_name": "Doe"},
        },
    ]


@pytest.fixture
def missing_metadata():
    return [
        {
            "name": "Alice",
            "addresses": [
                {
                    "number": 9562,
                    "street": "Morris St.",
                    "city": "Massillon",
                    "state": "OH",
                    "zip": 44646,
                }
            ],
        },
        {
            "addresses": [
                {
                    "number": 8449,
                    "street": "Spring St.",
                    "city": "Elizabethton",
                    "state": "TN",
                    "zip": 37643,
                }
            ]
        },
    ]


@pytest.fixture
def max_level_test_input_data():
    """
    input data to test json_normalize with max_level param
    """
    return [
        {
            "CreatedBy": {"Name": "User001"},
            "Lookup": {
                "TextField": "Some text",
                "UserField": {"Id": "ID001", "Name": "Name001"},
            },
            "Image": {"a": "b"},
        }
    ]


class TestJSONNormalize:
    def test_simple_records(self):
        recs = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5, "c": 6},
            {"a": 7, "b": 8, "c": 9},
            {"a": 10, "b": 11, "c": 12},
        ]

        result = json_normalize(recs)
        expected = DataFrame(recs)

        tm.assert_frame_equal(result, expected)

    # TODO(ArrayManager) sanitize S/U numpy dtypes to object
    @td.skip_array_manager_not_yet_implemented
    def test_simple_normalize(self, state_data):
        result = json_normalize(state_data[0], "counties")
        expected = DataFrame(state_data[0]["counties"])
        tm.assert_frame_equal(result, expected)

        result = json_normalize(state_data, "counties")

        expected = []
        for rec in state_data:
            expected.extend(rec["counties"])
        expected = DataFrame(expected)

        tm.assert_frame_equal(result, expected)

        result = json_normalize(state_data, "counties", meta="state")
        expected["state"] = np.array(["Florida", "Ohio"]).repeat([3, 2])

        tm.assert_frame_equal(result, expected)

    def test_empty_array(self):
        result = json_normalize([])
        expected = DataFrame()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "data, record_path, exception_type",
        [
            ([{"a": 0}, {"a": 1}], None, None),
            ({"a": [{"a": 0}, {"a": 1}]}, "a", None),
            ('{"a": [{"a": 0}, {"a": 1}]}', None, NotImplementedError),
            (None, None, NotImplementedError),
        ],
    )
    def test_accepted_input(self, data, record_path, exception_type):
        if exception_type is not None:
            with pytest.raises(exception_type, match=tm.EMPTY_STRING_PATTERN):
                json_normalize(data, record_path=record_path)
        else:
            result = json_normalize(data, record_path=record_path)
            expected = DataFrame([0, 1], columns=["a"])
            tm.assert_frame_equal(result, expected)

    def test_simple_normalize_with_separator(self, deep_nested):
        # GH 14883
        result = json_normalize({"A": {"A": 1, "B": 2}})
        expected = DataFrame([[1, 2]], columns=["A.A", "A.B"])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        result = json_normalize({"A": {"A": 1, "B": 2}}, sep="_")
        expected = DataFrame([[1, 2]], columns=["A_A", "A_B"])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        result = json_normalize({"A": {"A": 1, "B": 2}}, sep="\u03c3")
        expected = DataFrame([[1, 2]], columns=["A\u03c3A", "A\u03c3B"])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        result = json_normalize(
            deep_nested,
            ["states", "cities"],
            meta=["country", ["states", "name"]],
            sep="_",
        )
        expected = Index(["name", "pop", "country", "states_name"]).sort_values()
        assert result.columns.sort_values().equals(expected)

    def test_value_array_record_prefix(self):
        # GH 21536
        result = json_normalize({"A": [1, 2]}, "A", record_prefix="Prefix.")
        expected = DataFrame([[1], [2]], columns=["Prefix.0"])
        tm.assert_frame_equal(result, expected)

    def test_nested_object_record_path(self):
        # GH 22706
        data = {
            "state": "Florida",
            "info": {
                "governor": "Rick Scott",
                "counties": [
                    {"name": "Dade", "population": 12345},
                    {"name": "Broward", "population": 40000},
                    {"name": "Palm Beach", "population": 60000},
                ],
            },
        }
        result = json_normalize(data, record_path=["info", "counties"])
        expected = DataFrame(
            [["Dade", 12345], ["Broward", 40000], ["Palm Beach", 60000]],
            columns=["name", "population"],
        )
        tm.assert_frame_equal(result, expected)

    def test_more_deeply_nested(self, deep_nested):

        result = json_normalize(
            deep_nested, ["states", "cities"], meta=["country", ["states", "name"]]
        )
        ex_data = {
            "country": ["USA"] * 4 + ["Germany"] * 3,
            "states.name": [
                "California",
                "California",
                "Ohio",
                "Ohio",
                "Bayern",
                "Nordrhein-Westfalen",
                "Nordrhein-Westfalen",
            ],
            "name": [
                "San Francisco",
                "Los Angeles",
                "Columbus",
                "Cleveland",
                "Munich",
                "Duesseldorf",
                "Koeln",
            ],
            "pop": [12345, 12346, 1234, 1236, 12347, 1238, 1239],
        }

        expected = DataFrame(ex_data, columns=result.columns)
        tm.assert_frame_equal(result, expected)

    def test_shallow_nested(self):
        data = [
            {
                "state": "Florida",
                "shortname": "FL",
                "info": {"governor": "Rick Scott"},
                "counties": [
                    {"name": "Dade", "population": 12345},
                    {"name": "Broward", "population": 40000},
                    {"name": "Palm Beach", "population": 60000},
                ],
            },
            {
                "state": "Ohio",
                "shortname": "OH",
                "info": {"governor": "John Kasich"},
                "counties": [
                    {"name": "Summit", "population": 1234},
                    {"name": "Cuyahoga", "population": 1337},
                ],
            },
        ]

        result = json_normalize(
            data, "counties", ["state", "shortname", ["info", "governor"]]
        )
        ex_data = {
            "name": ["Dade", "Broward", "Palm Beach", "Summit", "Cuyahoga"],
            "state": ["Florida"] * 3 + ["Ohio"] * 2,
            "shortname": ["FL", "FL", "FL", "OH", "OH"],
            "info.governor": ["Rick Scott"] * 3 + ["John Kasich"] * 2,
            "population": [12345, 40000, 60000, 1234, 1337],
        }
        expected = DataFrame(ex_data, columns=result.columns)
        tm.assert_frame_equal(result, expected)

    def test_nested_meta_path_with_nested_record_path(self, state_data):
        # GH 27220
        result = json_normalize(
            data=state_data,
            record_path=["counties"],
            meta=["state", "shortname", ["info", "governor"]],
            errors="ignore",
        )

        ex_data = {
            "name": ["Dade", "Broward", "Palm Beach", "Summit", "Cuyahoga"],
            "population": [12345, 40000, 60000, 1234, 1337],
            "state": ["Florida"] * 3 + ["Ohio"] * 2,
            "shortname": ["FL"] * 3 + ["OH"] * 2,
            "info.governor": ["Rick Scott"] * 3 + ["John Kasich"] * 2,
        }

        expected = DataFrame(ex_data)
        tm.assert_frame_equal(result, expected)

    def test_meta_name_conflict(self):
        data = [
            {
                "foo": "hello",
                "bar": "there",
                "data": [
                    {"foo": "something", "bar": "else"},
                    {"foo": "something2", "bar": "else2"},
                ],
            }
        ]

        msg = r"Conflicting metadata name (foo|bar), need distinguishing prefix"
        with pytest.raises(ValueError, match=msg):
            json_normalize(data, "data", meta=["foo", "bar"])

        result = json_normalize(data, "data", meta=["foo", "bar"], meta_prefix="meta")

        for val in ["metafoo", "metabar", "foo", "bar"]:
            assert val in result

    def test_meta_parameter_not_modified(self):
        # GH 18610
        data = [
            {
                "foo": "hello",
                "bar": "there",
                "data": [
                    {"foo": "something", "bar": "else"},
                    {"foo": "something2", "bar": "else2"},
                ],
            }
        ]

        COLUMNS = ["foo", "bar"]
        result = json_normalize(data, "data", meta=COLUMNS, meta_prefix="meta")

        assert COLUMNS == ["foo", "bar"]
        for val in ["metafoo", "metabar", "foo", "bar"]:
            assert val in result

    # TODO(ArrayManager) sanitize S/U numpy dtypes to object
    @td.skip_array_manager_not_yet_implemented
    def test_record_prefix(self, state_data):
        result = json_normalize(state_data[0], "counties")
        expected = DataFrame(state_data[0]["counties"])
        tm.assert_frame_equal(result, expected)

        result = json_normalize(
            state_data, "counties", meta="state", record_prefix="county_"
        )

        expected = []
        for rec in state_data:
            expected.extend(rec["counties"])
        expected = DataFrame(expected)
        expected = expected.rename(columns=lambda x: "county_" + x)
        expected["state"] = np.array(["Florida", "Ohio"]).repeat([3, 2])

        tm.assert_frame_equal(result, expected)

    def test_non_ascii_key(self):
        testjson = (
            b'[{"\xc3\x9cnic\xc3\xb8de":0,"sub":{"A":1, "B":2}},'
            + b'{"\xc3\x9cnic\xc3\xb8de":1,"sub":{"A":3, "B":4}}]'
        ).decode("utf8")

        testdata = {
            b"\xc3\x9cnic\xc3\xb8de".decode("utf8"): [0, 1],
            "sub.A": [1, 3],
            "sub.B": [2, 4],
        }
        expected = DataFrame(testdata)

        result = json_normalize(json.loads(testjson))
        tm.assert_frame_equal(result, expected)

    def test_missing_field(self, author_missing_data):
        # GH20030:
        result = json_normalize(author_missing_data)
        ex_data = [
            {
                "info": np.nan,
                "info.created_at": np.nan,
                "info.last_updated": np.nan,
                "author_name.first": np.nan,
                "author_name.last_name": np.nan,
            },
            {
                "info": None,
                "info.created_at": "11/08/1993",
                "info.last_updated": "26/05/2012",
                "author_name.first": "Jane",
                "author_name.last_name": "Doe",
            },
        ]
        expected = DataFrame(ex_data)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "max_level,expected",
        [
            (
                0,
                [
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                ],
            ),
            (
                1,
                [
                    {
                        "TextField": "Some text",
                        "UserField.Id": "ID001",
                        "UserField.Name": "Name001",
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                    {
                        "TextField": "Some text",
                        "UserField.Id": "ID001",
                        "UserField.Name": "Name001",
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                ],
            ),
        ],
    )
    def test_max_level_with_records_path(self, max_level, expected):
        # GH23843: Enhanced JSON normalize
        test_input = [
            {
                "CreatedBy": {"Name": "User001"},
                "Lookup": [
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                    },
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                    },
                ],
                "Image": {"a": "b"},
                "tags": [
                    {"foo": "something", "bar": "else"},
                    {"foo": "something2", "bar": "else2"},
                ],
            }
        ]

        result = json_normalize(
            test_input,
            record_path=["Lookup"],
            meta=[["CreatedBy"], ["Image"]],
            max_level=max_level,
        )
        expected_df = DataFrame(data=expected, columns=result.columns.values)
        tm.assert_equal(expected_df, result)

    def test_nested_flattening_consistent(self):
        # see gh-21537
        df1 = json_normalize([{"A": {"B": 1}}])
        df2 = json_normalize({"dummy": [{"A": {"B": 1}}]}, "dummy")

        # They should be the same.
        tm.assert_frame_equal(df1, df2)

    def test_nonetype_record_path(self, nulls_fixture):
        # see gh-30148
        # should not raise TypeError
        result = json_normalize(
            [
                {"state": "Texas", "info": nulls_fixture},
                {"state": "Florida", "info": [{"i": 2}]},
            ],
            record_path=["info"],
        )
        expected = DataFrame({"i": 2}, index=[0])
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("value", ["false", "true", "{}", "1", '"text"'])
    def test_non_list_record_path_errors(self, value):
        # see gh-30148, GH 26284
        parsed_value = json.loads(value)
        test_input = {"state": "Texas", "info": parsed_value}
        test_path = "info"
        msg = (
            f"{test_input} has non list value {parsed_value} for path {test_path}. "
            "Must be list or null."
        )
        with pytest.raises(TypeError, match=msg):
            json_normalize([test_input], record_path=[test_path])

    def test_meta_non_iterable(self):
        # GH 31507
        data = """[{"id": 99, "data": [{"one": 1, "two": 2}]}]"""

        result = json_normalize(json.loads(data), record_path=["data"], meta=["id"])
        expected = DataFrame(
            {"one": [1], "two": [2], "id": np.array([99], dtype=object)}
        )
        tm.assert_frame_equal(result, expected)

    def test_generator(self, state_data):
        # GH35923 Fix pd.json_normalize to not skip the first element of a
        # generator input
        def generator_data():
            yield from state_data[0]["counties"]

        result = json_normalize(generator_data())
        expected = DataFrame(state_data[0]["counties"])

        tm.assert_frame_equal(result, expected)


class TestNestedToRecord:
    def test_flat_stays_flat(self):
        recs = [{"flat1": 1, "flat2": 2}, {"flat3": 3, "flat2": 4}]
        result = nested_to_record(recs)
        expected = recs
        assert result == expected

    def test_one_level_deep_flattens(self):
        data = {"flat1": 1, "dict1": {"c": 1, "d": 2}}

        result = nested_to_record(data)
        expected = {"dict1.c": 1, "dict1.d": 2, "flat1": 1}

        assert result == expected

    def test_nested_flattens(self):
        data = {
            "flat1": 1,
            "dict1": {"c": 1, "d": 2},
            "nested": {"e": {"c": 1, "d": 2}, "d": 2},
        }

        result = nested_to_record(data)
        expected = {
            "dict1.c": 1,
            "dict1.d": 2,
            "flat1": 1,
            "nested.d": 2,
            "nested.e.c": 1,
            "nested.e.d": 2,
        }

        assert result == expected

    def test_json_normalize_errors(self, missing_metadata):
        # GH14583:
        # If meta keys are not always present a new option to set
        # errors='ignore' has been implemented

        msg = "Try running with errors='ignore' as key 'name' is not always present"
        with pytest.raises(KeyError, match=msg):
            json_normalize(
                data=missing_metadata,
                record_path="addresses",
                meta="name",
                errors="raise",
            )

    def test_missing_meta(self, missing_metadata):
        # GH25468
        # If metadata is nullable with errors set to ignore, the null values
        # should be numpy.nan values
        result = json_normalize(
            data=missing_metadata, record_path="addresses", meta="name", errors="ignore"
        )
        ex_data = [
            [9562, "Morris St.", "Massillon", "OH", 44646, "Alice"],
            [8449, "Spring St.", "Elizabethton", "TN", 37643, np.nan],
        ]
        columns = ["city", "number", "state", "street", "zip", "name"]
        columns = ["number", "street", "city", "state", "zip", "name"]
        expected = DataFrame(ex_data, columns=columns)
        tm.assert_frame_equal(result, expected)

    def test_donot_drop_nonevalues(self):
        # GH21356
        data = [
            {"info": None, "author_name": {"first": "Smith", "last_name": "Appleseed"}},
            {
                "info": {"created_at": "11/08/1993", "last_updated": "26/05/2012"},
                "author_name": {"first": "Jane", "last_name": "Doe"},
            },
        ]
        result = nested_to_record(data)
        expected = [
            {
                "info": None,
                "author_name.first": "Smith",
                "author_name.last_name": "Appleseed",
            },
            {
                "author_name.first": "Jane",
                "author_name.last_name": "Doe",
                "info.created_at": "11/08/1993",
                "info.last_updated": "26/05/2012",
            },
        ]

        assert result == expected

    def test_nonetype_top_level_bottom_level(self):
        # GH21158: If inner level json has a key with a null value
        # make sure it does not do a new_d.pop twice and except
        data = {
            "id": None,
            "location": {
                "country": {
                    "state": {
                        "id": None,
                        "town.info": {
                            "id": None,
                            "region": None,
                            "x": 49.151580810546875,
                            "y": -33.148521423339844,
                            "z": 27.572303771972656,
                        },
                    }
                }
            },
        }
        result = nested_to_record(data)
        expected = {
            "id": None,
            "location.country.state.id": None,
            "location.country.state.town.info.id": None,
            "location.country.state.town.info.region": None,
            "location.country.state.town.info.x": 49.151580810546875,
            "location.country.state.town.info.y": -33.148521423339844,
            "location.country.state.town.info.z": 27.572303771972656,
        }
        assert result == expected

    def test_nonetype_multiple_levels(self):
        # GH21158: If inner level json has a key with a null value
        # make sure it does not do a new_d.pop twice and except
        data = {
            "id": None,
            "location": {
                "id": None,
                "country": {
                    "id": None,
                    "state": {
                        "id": None,
                        "town.info": {
                            "region": None,
                            "x": 49.151580810546875,
                            "y": -33.148521423339844,
                            "z": 27.572303771972656,
                        },
                    },
                },
            },
        }
        result = nested_to_record(data)
        expected = {
            "id": None,
            "location.id": None,
            "location.country.id": None,
            "location.country.state.id": None,
            "location.country.state.town.info.region": None,
            "location.country.state.town.info.x": 49.151580810546875,
            "location.country.state.town.info.y": -33.148521423339844,
            "location.country.state.town.info.z": 27.572303771972656,
        }
        assert result == expected

    @pytest.mark.parametrize(
        "max_level, expected",
        [
            (
                None,
                [
                    {
                        "CreatedBy.Name": "User001",
                        "Lookup.TextField": "Some text",
                        "Lookup.UserField.Id": "ID001",
                        "Lookup.UserField.Name": "Name001",
                        "Image.a": "b",
                    }
                ],
            ),
            (
                0,
                [
                    {
                        "CreatedBy": {"Name": "User001"},
                        "Lookup": {
                            "TextField": "Some text",
                            "UserField": {"Id": "ID001", "Name": "Name001"},
                        },
                        "Image": {"a": "b"},
                    }
                ],
            ),
            (
                1,
                [
                    {
                        "CreatedBy.Name": "User001",
                        "Lookup.TextField": "Some text",
                        "Lookup.UserField": {"Id": "ID001", "Name": "Name001"},
                        "Image.a": "b",
                    }
                ],
            ),
        ],
    )
    def test_with_max_level(self, max_level, expected, max_level_test_input_data):
        # GH23843: Enhanced JSON normalize
        output = nested_to_record(max_level_test_input_data, max_level=max_level)
        assert output == expected

    def test_with_large_max_level(self):
        # GH23843: Enhanced JSON normalize
        max_level = 100
        input_data = [
            {
                "CreatedBy": {
                    "user": {
                        "name": {"firstname": "Leo", "LastName": "Thomson"},
                        "family_tree": {
                            "father": {
                                "name": "Father001",
                                "father": {
                                    "Name": "Father002",
                                    "father": {
                                        "name": "Father003",
                                        "father": {"Name": "Father004"},
                                    },
                                },
                            }
                        },
                    }
                }
            }
        ]
        expected = [
            {
                "CreatedBy.user.name.firstname": "Leo",
                "CreatedBy.user.name.LastName": "Thomson",
                "CreatedBy.user.family_tree.father.name": "Father001",
                "CreatedBy.user.family_tree.father.father.Name": "Father002",
                "CreatedBy.user.family_tree.father.father.father.name": "Father003",
                "CreatedBy.user.family_tree.father.father.father.father.Name": "Father004",  # noqa: E501
            }
        ]
        output = nested_to_record(input_data, max_level=max_level)
        assert output == expected

    def test_deprecated_import(self):
        with tm.assert_produces_warning(FutureWarning):
            from pandas.io.json import json_normalize

            recs = [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]
            json_normalize(recs)

    def test_series_non_zero_index(self):
        # GH 19020
        data = {
            0: {"id": 1, "name": "Foo", "elements": {"a": 1}},
            1: {"id": 2, "name": "Bar", "elements": {"b": 2}},
            2: {"id": 3, "name": "Baz", "elements": {"c": 3}},
        }
        s = Series(data)
        s.index = [1, 2, 3]
        result = json_normalize(s)
        expected = DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Foo", "Bar", "Baz"],
                "elements.a": [1.0, np.nan, np.nan],
                "elements.b": [np.nan, 2.0, np.nan],
                "elements.c": [np.nan, np.nan, 3.0],
            }
        )
        tm.assert_frame_equal(result, expected)
