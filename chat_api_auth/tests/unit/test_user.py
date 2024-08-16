import pytest

from chat_api_auth.django_auth.user import split_name


@pytest.mark.parametrize(
    "param_name, expected_outcome",
    [
        ["Janusz Kowalski", ("Janusz", "Kowalski")],
        ["Anna Maria Wesołowska", ("Anna Maria", "Wesołowska")],
        ["Janusz", ("Janusz", "")],
        ["", ("", "")],
        [None, ("", "")],
    ],
)
def test_split_name(param_name, expected_outcome):
    result = split_name(param_name)

    assert result == expected_outcome
