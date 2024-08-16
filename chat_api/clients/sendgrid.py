import os
import typing

from sendgrid import Mail, SendGridAPIClient, To
from chat_api.clients.firebase import FirebaseClient

import settings


class SendGridEmailClient:
    def __init__(self):
        self.sendgrid_client = SendGridAPIClient(api_key=settings.SENDGRID_CONFIG["API_KEY"])
        self.firebase_client = FirebaseClient()

    def send_email_from_template(
        self,
        to: str,
        subject: str,
        template_id: str,
        dynamic_template_data: typing.Dict,
    ):
        message = Mail(
            from_email=(settings.SENDGRID_FROM_EMAIL, "chat-api App"),
            to_emails=To(email=to),
            subject=subject,
        )

        message.template_id = template_id
        message.dynamic_template_data = dynamic_template_data

        self.sendgrid_client.send(message)
