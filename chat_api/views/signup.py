from chat_api_auth.django_auth.authentication import ChatApiDjangoAuthenticationToken
from rest_framework import generics
from rest_framework.response import Response

from chat_api.exceptions.user_creation_exception import UserCreationException
from chat_api.repositories.signup import SignupRepository
from chat_api.serializers.signup import SignupSerializer, SSOSignupSerializer


class SignupView(generics.CreateAPIView):
    serializer_class = SignupSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        validated_data = serializer.validated_data
        try:
            SignupRepository().signup(validated_data)
            return Response(status=201)
        except UserCreationException as exc:
            return Response(exc.args[0], status=400)


class SignUpWithSSOView(generics.GenericAPIView):
    authentication_classes = (ChatApiDjangoAuthenticationToken,)
    serializer_class = SSOSignupSerializer

    def post(self, request, *args, **kwargs):
        user = request.user

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        invitation = serializer.validated_data.get("user_invitation")

        return SignupRepository().signup_with_sso(user, invitation)
