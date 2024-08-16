from firebase_admin import auth
from firebase_admin.auth import TokenSignError

from chat_api.clients.firebase import FirebaseClient, FirebaseUserGetException
from chat_api.exceptions.user.user_impersonation import ImpersonateUserException
import settings


class ImpersonateUserService:
    @staticmethod
    def _check_firebase_uid_exists(firebase_uid):
        FirebaseClient.get_user_by_uid(firebase_uid)

    def generate_impersonate_token(self, firebase_uid):
        custom_claim = {"isImpersonated": True}
        try:
            self._check_firebase_uid_exists(firebase_uid)
            token = auth.create_custom_token(firebase_uid, custom_claim)
            return token.decode("utf-8")
        except (TokenSignError, FirebaseUserGetException, ValueError) as exc:
            raise ImpersonateUserException(exc)

    def generate_redirection_link(self, firebase_uid):
        custom_token = self.generate_impersonate_token(firebase_uid)
        return f"{settings.GC_REACT_APP_URL}impersonate?token={custom_token}"
