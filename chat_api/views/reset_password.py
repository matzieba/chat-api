from rest_framework import generics
from rest_framework.response import Response

from chat_api.models import User
from chat_api.serializers.reset_password import ResetPasswordSerializer
from chat_api.services.email import EmailService


class ResetPasswordView(generics.GenericAPIView):
    serializer_class = ResetPasswordSerializer

    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            email_service = EmailService()
            email_service.reset_password_email(
                User.objects.get(email=serializer.validated_data["email"])
            )
        except User.DoesNotExist:
            return Response(status=404)
        return Response(status=201)
