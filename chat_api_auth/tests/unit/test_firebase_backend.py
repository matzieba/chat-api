import pytest

from chat_api_auth.backends.firebase import ChatApiAuthenticationError


class TestFirebaseJWTBackend:
    def test_authenticate(self, firebase_backend, firebase_accepts, user):
        result = firebase_backend.authenticate("Bearer correct_token")

        assert result == user

    def test_firebase_returns_no_user(self, firebase_backend, firebase_returns_no_user):
        with pytest.raises(ChatApiAuthenticationError) as exc:
            firebase_backend.authenticate("Bearer correct_token")

        assert exc.value.description == "No user in JWT Token"

    def test_firebase_rejects(self, firebase_backend, firebase_rejects):
        with pytest.raises(ChatApiAuthenticationError) as exc:
            firebase_backend.authenticate("Bearer correct_token")

        assert exc.value.description == "Invalid JWT Credentials: Token Expired"
