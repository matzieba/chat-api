import re
from typing import Any, Mapping, Union

from django.contrib.auth import get_user_model
from django.contrib.auth.models import AbstractUser

from chat_api_auth.backends.exceptions import ChatApiAuthenticationError


def user_loader(user: Mapping) -> Union[Any, AbstractUser]:
    if not user:
        raise ChatApiAuthenticationError("Missing user in token")

    User = get_user_model()

    try:
        return User.objects.get(firebase_uid=user["user_id"])
    except User.DoesNotExist:
        first_name, last_name = split_name(user.get("name"))
        user_instance = User(
            firebase_uid=user["user_id"], email=user.get("email", None)
        )
        user_instance.is_internal = False
        user_instance.first_name = first_name
        user_instance.last_name = last_name
        return user_instance


def split_name(name):
    if not name:
        return "", ""

    splitted_name = name.rsplit(" ", 1)
    if len(splitted_name) == 2:
        return splitted_name[0], splitted_name[1]
    elif len(splitted_name) == 1:
        return splitted_name[0], ""
    else:
        return "", ""


def internal_user_loader(user: Mapping) -> Union[Any, AbstractUser]:
    if not user:
        return None

    User = get_user_model()
    user_instance = User(firebase_uid=user["service_name"])
    user_instance.is_internal = True

    return user_instance
