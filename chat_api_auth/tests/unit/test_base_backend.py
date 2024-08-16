from unittest import mock

import pytest

from chat_api_auth.backends.base import BaseBackend
from chat_api_auth.backends.firebase import ChatApiAuthenticationError


def mock_get_auth_header_prefix(_):
    return "Bearer"


@pytest.mark.parametrize(
    "param_header, expected_error",
    [
        (None, "Missing Authorization Header"),
        ("NotBearer tttttt", "Invalid Authorization Header: Must start with Bearer"),
        ("BearerNoSpaceToken", "Invalid Authorization Header: Token Missing"),
        (
            "Bearer Too Many Spaces",
            "Invalid Authorization Header: Contains extra content",
        ),
    ],
)
def test_parsing_token_from_request(default_user_loader, param_header, expected_error):
    with mock.patch.object(
        BaseBackend, "get_auth_header_prefix", return_value="Bearer"
    ):
        with pytest.raises(ChatApiAuthenticationError) as exc:
            backend = BaseBackend(default_user_loader)
            backend.get_auth_header_prefix = mock_get_auth_header_prefix.__get__(
                backend
            )
            BaseBackend(default_user_loader).parse_auth_token_from_request(param_header)

        assert exc.value.description == expected_error
