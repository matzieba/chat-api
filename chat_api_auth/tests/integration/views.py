from rest_framework import views
from rest_framework.response import Response

from chat_api_auth.django_auth.authentication import (
    ChatApiDjangoAuthenticationInternal,
    ChatApiDjangoAuthenticationInternalOptional,
    ChatApiDjangoAuthenticationToken,
    ChatApiDjangoAuthenticationTokenOptional,
)


class AuthenticatedView(views.APIView):
    authentication_classes = (ChatApiDjangoAuthenticationToken,)

    def get(self, request, format=None):
        return Response(request.user.firebase_uid)


class UnAuthenticatedView(views.APIView):
    permission_classes = ()
    authentication_classes = ()

    def get(self, request, format=None):
        return Response("hello_world")


class AuthenticatedInternalView(views.APIView):
    authentication_classes = (ChatApiDjangoAuthenticationInternal,)
    permission_classes = ()

    def get(self, request, format=None):
        return Response(request.user.firebase_uid)


class AuthenticatedOptionalView(views.APIView):
    authentication_classes = (ChatApiDjangoAuthenticationTokenOptional,)
    permission_classes = ()

    def get(self, request, format=None):
        if request.user.is_anonymous:
            return Response()
        return Response(request.user.firebase_uid)


class AuthenticatedOptionalInternalAndTokenView(views.APIView):
    authentication_classes = (
        ChatApiDjangoAuthenticationInternalOptional,
        ChatApiDjangoAuthenticationToken,
    )

    def get(self, request, format=None):
        return Response(request.user.firebase_uid)
