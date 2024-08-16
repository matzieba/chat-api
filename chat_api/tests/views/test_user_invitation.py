from unittest import mock

import pytest

from chat_api.models import User
from chat_api.models.user_invitation import UserInvitation


@pytest.fixture
def send_user_invitation_email_mock():
    with mock.patch(
        "chat_api.views.user_invitation.EmailService.send_user_invitation_email"
    ) as send_user_invitation_email:
        yield send_user_invitation_email


def test_create_user_invitation_unauthorized(
    db,
    rest_client,
):
    response = rest_client.post(
        f"/chat-api/v1/api/user-invitations/",
        format="json",
    )
    assert response.status_code == 401


def test_create_user_invitation_invalid_payload(
    db,
    auth_client,
    user_with_company,
):
    response = auth_client(user_with_company).post(
        f"/chat-api/v1/api/user-invitations/", format="json", data={"a": "b"}
    )
    assert response.status_code == 400
    assert response.json() == {"email": ["This field is required."]}


def test_create_user_invitation_success(
    db, auth_client, user_with_company, send_user_invitation_email_mock
):
    email = "inv@cvt.services"
    response = auth_client(user_with_company).post(
        f"/chat-api/v1/api/user-invitations/", format="json", data={"email": email}
    )
    assert response.status_code == 201
    invitation_id = response.json().get("invitation_id")
    invitation = UserInvitation.objects.get(pk=invitation_id)

    assert invitation.email == email
    assert invitation.invited_by == user_with_company
    assert invitation.company == user_with_company.company

    assert send_user_invitation_email_mock.call_args[0][0] == invitation


def test_create_user_invitation_already_existing_user(
    db, auth_client, user_with_company, send_user_invitation_email_mock
):
    email = "inv@cvt.services"
    User(firebase_uid="123", email=email, company=user_with_company.company).save()

    response = auth_client(user_with_company).post(
        f"/chat-api/v1/api/user-invitations/", format="json", data={"email": email}
    )
    assert response.status_code == 400
    assert response.json() == {"user": ["User already exists"]}


def test_get_user_invitation_unauthorized_success(db, rest_client, user_with_company):
    invitation = UserInvitation(
        email="inv@cvt.services",
        company=user_with_company.company,
        invited_by=user_with_company,
    )
    invitation.save()

    response = rest_client.get(
        f"/chat-api/v1/api/user-invitations/{invitation.id}/",
        format="json",
    )
    assert response.status_code == 200
