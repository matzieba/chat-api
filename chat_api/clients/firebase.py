import secrets

from firebase_admin import auth
from firebase_admin.exceptions import FirebaseError


class FirebaseUserCreateException(Exception):
    pass


class FirebaseUserDeleteException(Exception):
    pass


class FirebaseUserGetException(Exception):
    pass

class FirebaseUserUpdateException(Exception):
    pass

class FirebaseClient:
    @staticmethod
    def create_firebase_user(name, surname, email):
        try:
            firebase_user = auth.create_user(
                display_name=f"{name} {surname}",
                email=email,
                password=secrets.token_urlsafe(32),
            )
            return firebase_user.uid
        except FirebaseError as err:
            raise FirebaseUserCreateException(err.code)

    @staticmethod
    def get_user_by_email(email):
        try:
            firebase_user = auth.get_user_by_email(
                email=email,
            )
            return firebase_user.uid
        except FirebaseError as err:
            raise FirebaseUserGetException(err.code)

    @staticmethod
    def get_user_by_uid(uid):
        try:
            firebase_user = auth.get_user(
                uid=uid,
            )
            return firebase_user
        except FirebaseError as err:
            raise FirebaseUserGetException(err.code)

    @staticmethod
    def create_firebase_user_with_username_and_password(email, password):
        try:
            firebase_user = auth.create_user(
                email=email,
                password=password,
            )
            return firebase_user.uid
        except FirebaseError as err:
            raise FirebaseUserCreateException(err.code)

    @staticmethod
    def delete_firebase_user(firebase_uid):
        try:
            auth.delete_user(firebase_uid)
        except FirebaseError as err:
            raise FirebaseUserDeleteException(err.code)

    @staticmethod
    def logout_from_all_sessions(firebase_uid):
        auth.revoke_refresh_tokens(firebase_uid)

    @staticmethod
    def generate_reset_password_link(email):
        return auth.generate_password_reset_link(email)

    @staticmethod
    def update_firebase_user(firebase_uid, **kwargs):
        try:
            return auth.update_user(firebase_uid, **kwargs)
        except (FirebaseError, ValueError):
            raise FirebaseUserUpdateException()
