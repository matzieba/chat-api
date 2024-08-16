import base64

from django.test import override_settings
import pytest

from chat_api_auth.backends.firebase import ChatApiAuthenticationError

AUTH_CREDENTIALS = {
    "internal_services": {
        "test_service": "test_service_password",
    },
}


class TestInternalBackend:
    @override_settings(AUTH=AUTH_CREDENTIALS)
    def test_authenticate(self, internal_backend, internal_user):
        result = internal_backend.authenticate(
            "Basic "
            + self._encode_username_and_password(
                "test_service", "test_service_password"
            )
        )

        assert result == internal_user

    @override_settings(AUTH=AUTH_CREDENTIALS)
    def test_invalid_username(self, internal_backend):
        with pytest.raises(ChatApiAuthenticationError) as exc:
            internal_backend.authenticate(
                "Basic "
                + self._encode_username_and_password("invalid", "test_service_password")
            )

        assert exc.value.description == "Invalid Authorization Header: Token Invalid"

    @override_settings(AUTH=AUTH_CREDENTIALS)
    def test_invalid_password(self, internal_backend):
        with pytest.raises(ChatApiAuthenticationError) as exc:
            internal_backend.authenticate(
                "Basic " + self._encode_username_and_password("test_service", "invalid")
            )

        assert exc.value.description == "Invalid Authorization Header: Token Invalid"

    @override_settings(AUTH=AUTH_CREDENTIALS)
    def test_invalid_base64(self, internal_backend):
        with pytest.raises(ChatApiAuthenticationError) as exc:
            internal_backend.authenticate(
                "Basic a2HFgmFiYW5nYQ=="
            ) 
            
        assert exc.value.description == "Invalid Authorization Header: Token Invalid"

    @override_settings(AUTH=AUTH_CREDENTIALS)
    def test_invalid_base64(self, internal_backend):
        with pytest.raises(ChatApiAuthenticationError) as exc:
            internal_backend.authenticate("Basic ó")

        assert exc.value.description == "Invalid Authorization Header: Token Invalid"

    @override_settings(AUTH=AUTH_CREDENTIALS)
    def _encode_username_and_password(self, username: str, password: str):
        credentials = username + ":" + password
        return base64.b64encode(bytes(credentials.encode())).decode("utf-8")
