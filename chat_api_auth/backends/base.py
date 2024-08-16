from typing import Any, Callable, Mapping

from chat_api_auth.backends.exceptions import ChatApiAuthenticationError


class BaseBackend:
    def __init__(self, user_loader: Callable[[Mapping], Any]):
        self.user_loader = user_loader

    def parse_auth_token_from_request(self, auth_header: str) -> str:
        """Parses and returns Auth token from the request header. Raises `ChatApiAuthenticationError exception`"""
        if not auth_header:
            raise ChatApiAuthenticationError(description="Missing Authorization Header")

        parts = auth_header.split()

        if len(parts) == 1:
            raise ChatApiAuthenticationError(
                description="Invalid Authorization Header: Token Missing"
            )
        elif parts[0].lower() != self.get_auth_header_prefix().lower():
            raise ChatApiAuthenticationError(
                description="Invalid Authorization Header: Must start with {0}".format(
                    self.get_auth_header_prefix()
                )
            )
        elif len(parts) > 2:
            raise ChatApiAuthenticationError(
                description="Invalid Authorization Header: Contains extra content"
            )

        return parts[1]

    def authenticate(self, auth_header: str):
        raise NotImplementedError

    def get_auth_header_prefix(self):
        raise NotImplementedError
