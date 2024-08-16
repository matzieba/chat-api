from rest_framework import exceptions
from rest_framework.authentication import BaseAuthentication, get_authorization_header

from chat_api_auth.backends.base import BaseBackend
from chat_api_auth.backends.firebase import FirebaseJWTBackend, ChatApiAuthenticationError
from chat_api_auth.backends.internal import InternalBackend

from .user import internal_user_loader, user_loader


def auth_generator(backend):
    class ChatApiDjangoAuthentication(BaseAuthentication):
        backend: BaseBackend

        def authenticate(self, request):
            """
            Returns a two-tuple of `User` and token if a valid signature has been
            supplied using provided backend authentication.  Otherwise returns `None`.
            """
            auth_header = get_authorization_header(request)
            auth_header = auth_header.decode("utf-8")

            try:
                user = self.backend.authenticate(auth_header)
                return user, None
            except ChatApiAuthenticationError as exc:
                raise exceptions.AuthenticationFailed(detail=exc.description)

        def authenticate_header(self, request):
            return 'Bearer realm="Access to the ChatApi api"'

    ChatApiDjangoAuthentication.backend = backend
    return ChatApiDjangoAuthentication


def optional_auth_generator(backend):
    class ChatApiDjangoOptionalAuthentication(BaseAuthentication):
        backend: BaseBackend

        def authenticate(self, request):
            """
            Returns a two-tuple of `User` and token if a valid signature has been
            supplied using provided backend authentication.  Otherwise returns `None`.
            Ignores exceptions and allows chaining multiple authentication backends
            """
            auth_header = get_authorization_header(request)
            auth_header = auth_header.decode("utf-8")

            try:
                user = self.backend.authenticate(auth_header)
                return user, None
            except ChatApiAuthenticationError as exc:
                return None

    ChatApiDjangoOptionalAuthentication.backend = backend
    return ChatApiDjangoOptionalAuthentication


ChatApiDjangoAuthenticationToken = auth_generator(FirebaseJWTBackend(user_loader))
ChatApiDjangoAuthenticationTokenOptional = optional_auth_generator(
    FirebaseJWTBackend(user_loader)
)
ChatApiDjangoAuthenticationInternal = auth_generator(InternalBackend(internal_user_loader))
ChatApiDjangoAuthenticationInternalOptional = optional_auth_generator(
    InternalBackend(internal_user_loader)
)
