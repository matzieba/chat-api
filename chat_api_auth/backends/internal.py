import base64
from typing import Any

from django.conf import settings

from chat_api_auth.backends.base import BaseBackend
from chat_api_auth.backends.exceptions import ChatApiAuthenticationError


class InternalBackend(BaseBackend):
    def authenticate(self, auth_header: str) -> Any:
        """
        Extract auth token from `authorization` header, decode jwt token, verify firebase and return either a ``user``
        object if successful else raise an `ChatApiAuthenticationError` exception
        """
        encoded_credentials = self.parse_auth_token_from_request(auth_header)

        try:
            credentials = (
                base64.b64decode(encoded_credentials).decode("utf-8").split(":")
            )
        except Exception:
            raise ChatApiAuthenticationError(
                description="Invalid Authorization Header: Token Invalid"
            )

        services_credentials = settings.AUTH["internal_services"]

        if (
            credentials[0] in services_credentials
            and services_credentials[credentials[0]] == credentials[1]
        ):
            return self.user_loader({"service_name": credentials[0]})

        raise ChatApiAuthenticationError(
            description="Invalid Authorization Header: Token Invalid"
        )

    def get_auth_header_prefix(self):
        return "Basic"
