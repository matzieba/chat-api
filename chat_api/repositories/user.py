from django.core.exceptions import ObjectDoesNotExist

from chat_api.exceptions.user_does_not_exist import UserDoesNotExist
from chat_api.models import User


class UserRepository:
    @staticmethod
    def get_queryset():
        return User.objects.prefetch_related("company")

    @staticmethod
    def get_me(firebase_uid):
        try:
            user_instance = UserRepository.get_queryset().get(firebase_uid=firebase_uid)
            return user_instance
        except ObjectDoesNotExist as err:
            raise UserDoesNotExist(err.args[0])
