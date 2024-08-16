from rest_framework import mixins, viewsets
from rest_framework.response import Response

from chat_api.models.user_invitation import UserInvitation
from chat_api.serializers.user_invitation import (
    CreateUserInvitationSerializer,
    UserInvitationSerializer,
)
from chat_api.services.email import EmailService
from chat_api_auth.django_auth.authentication import ChatApiDjangoAuthenticationToken


class UserInvitationViewSet(
    viewsets.GenericViewSet, mixins.CreateModelMixin, mixins.RetrieveModelMixin
):
    authentication_classes = (ChatApiDjangoAuthenticationToken,)
    permission_classes = ()
    serializer_class = CreateUserInvitationSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        invitation = UserInvitation(
            email=serializer.validated_data["email"],
            company=request.user.company,
            invited_by=request.user,
        )
        invitation.save()

        EmailService().send_user_invitation_email(invitation)
        return Response(status=201, data={"invitation_id": invitation.id})

    def retrieve(self, request, *args, **kwargs):
        self.serializer_class = UserInvitationSerializer

        return super().retrieve(request, *args, **kwargs)

    def get_queryset(self):
        return UserInvitation.objects.all()

    def get_authenticators(self):
        if self.request.method == "GET":
            return ()
        return super().get_authenticators()
