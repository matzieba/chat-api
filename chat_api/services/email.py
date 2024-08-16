from chat_api.clients.sendgrid import SendGridEmailClient
from chat_api.models.user_invitation import UserInvitation
import settings
from chat_api.clients.firebase import FirebaseClient
from chat_api.models import User

class EmailService:
    def __init__(self):
        self.email_client = SendGridEmailClient()
        self.firebase_client = FirebaseClient

    def send_user_invitation_email(self, user_invitation: UserInvitation):
        invitation_link = f"{settings.REACT_APP_URL}invitation/?id={user_invitation.id}"

        self.email_client.send_email_from_template(
            to=user_invitation.email,
            subject=f"{user_invitation.company.name} invited you to join the Workspace!",
            template_id=settings.SENDGRID_CONFIG["TEMPLATES"]["INVITATION"],
            dynamic_template_data={
                "email": user_invitation.email,
                "invitation_link": invitation_link,
                "company_name": user_invitation.company.name,
            },
        )

    def reset_password_email(self, user: User):
        self.email_client.send_email_from_template(
            to=user.email,
            subject="SalesPlaybook reset password",
            template_id=settings.SENDGRID_CONFIG["TEMPLATES"]["RESET_PASSWORD"],
            dynamic_template_data={
                "email": user.email,
                "first_name_last_name": f"{user.first_name} {user.last_name}",
                "reset_link": self.firebase_client.generate_reset_password_link(
                    user.email
                ),
            },
        )
