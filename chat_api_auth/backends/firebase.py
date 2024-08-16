from typing import Any

from firebase_admin import auth as firebase_auth
from firebase_admin.auth import (
    CertificateFetchError,
    ExpiredIdTokenError,
    InvalidIdTokenError,
    RevokedIdTokenError,
)

from chat_api_auth.backends.base import BaseBackend
from chat_api_auth.backends.exceptions import ChatApiAuthenticationError


class FirebaseJWTBackend(BaseBackend):
    def authenticate(self, auth_header: str) -> Any:
        """
        Extract auth token from `authorization` header, decode jwt token, verify firebase and return either a ``user``
        object if successful else raise an `ChatApiAuthenticationError` exception
        """
        token = self.parse_auth_token_from_request(auth_header)
        try:
            decoded_token = firebase_auth.verify_id_token(token)
        except (
            ValueError,
            InvalidIdTokenError,
            ExpiredIdTokenError,
            RevokedIdTokenError,
            CertificateFetchError,
        ) as exc:
            raise ChatApiAuthenticationError(
                description=f"Invalid JWT Credentials: {exc.args[0]}"
            )

        user = self.user_loader(decoded_token)
        if not user:
            raise ChatApiAuthenticationError(description="No user in JWT Token")

        return user

    def get_auth_header_prefix(self):
        return "Bearer"
