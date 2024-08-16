from chat_api.models.user_invitation import UserInvitation
from chat_api.exceptions.user_invitation import UserInvitationDoesNotExistException
from django.core.exceptions import ObjectDoesNotExist


class UserInvitationRepository:
    @staticmethod
    def get_user_invitation(id: str):
        try:
            return UserInvitation.objects.get(id=id)
        except ObjectDoesNotExist:
            raise UserInvitationDoesNotExistException()

    @staticmethod
    def deactivate(instance: UserInvitation):
        instance.status = "accepted"
        instance.save()