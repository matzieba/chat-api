from django.core.exceptions import ObjectDoesNotExist
from rest_framework.response import Response

from chat_api.clients.firebase import FirebaseClient, FirebaseUserCreateException
from chat_api.exceptions.user_creation_exception import UserCreationException
from chat_api.exceptions.user_duplicated_email_exception import (
    UserDuplicatedEmailException,
)
from chat_api.models import Company, User, UserInvitation
from chat_api.repositories.user_invitation import UserInvitationRepository


class SignupRepository:
    def signup(self, user_data):
        try:
            if User.objects.filter(email=user_data["email"]):
                raise UserDuplicatedEmailException(
                    "User with this email already exists in database"
                )

            firebase_uid = (
                FirebaseClient().create_firebase_user_with_username_and_password(
                    user_data["email"], user_data["password"]
                )
            )

            user_instance, created = User.objects.get_or_create(
                firebase_uid=firebase_uid
            )
            user_instance.email = user_data["email"]
            user_instance.first_name = user_data["first_name"]
            user_instance.last_name = user_data["last_name"]
            user_instance.phone = user_data.get("phone")
            user_instance.job_title = user_data.get("job_title")

            invitation = user_data.get("user_invitation")
            if invitation:
                company = invitation.company
                user_instance.company = company

                UserInvitationRepository.deactivate(invitation)
            else:
                if user_data.get("company"):
                    if not isinstance(user_data["company"], Company):
                        company = Company()
                        company.name = user_data["company"]
                        company.save()
                    else:
                        company = user_data["company"]

                    user_instance.company = company

            user_instance.save()
            return user_instance
        except (FirebaseUserCreateException, UserDuplicatedEmailException) as exc:
            raise UserCreationException(exc.args[0])

    def signup_with_sso(self, user: User, user_invitation: UserInvitation):
        try:
            User.objects.get(firebase_uid=user.firebase_uid)
            return Response(status=200)

        except ObjectDoesNotExist:
            existing_user = User.objects.filter(email=user.email).first()

            if not existing_user:
                if user_invitation:
                    company = user_invitation.company
                    UserInvitationRepository.deactivate(user_invitation)
                else:
                    company = Company(
                        name=self._generate_company_name_from_sso_user(user)
                    )
                    company.save()

                User.objects.get_or_create(
                    firebase_uid=user.firebase_uid,
                    defaults={
                        "email": user.email,
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "company": company,
                    },
                )
                return Response(status=201)
            else:
                return Response(status=400)

    def _generate_company_name_from_sso_user(self, user):
        if user.first_name or user.last_name:
            name = " ".join([user.first_name or "", user.last_name or ""]).strip()
            return f"{name}'s Personal Company"
        else:
            return "Company Name"
