from rest_framework import generics
from rest_framework.response import Response

from chat_api.serializers.reset_password import ResetPasswordSerializer


class ResetPasswordView(generics.GenericAPIView):
    serializer_class = ResetPasswordSerializer

    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return Response(status=201)
