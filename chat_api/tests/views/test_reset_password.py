from unittest import mock

import pytest

from chat_api.services.email import EmailService


@pytest.fixture()
def mock_resetting_password():
    with mock.patch.object(
        EmailService, "reset_password_email", autospec=True
    ) as mocked_reset_password:
        yield mocked_reset_password


def test_reset_password_for_existing_user(
    db, user_with_company, rest_client, mock_resetting_password
):
    response = rest_client.post(
        "/chat-api/v1/api/users/reset-password/",
        data={"email": user_with_company.email},
        format="json",
    )

    assert response.status_code == 201
    mock_resetting_password.assert_called_once_with(mock.ANY, user_with_company)


def test_reset_password_for_not_existing_user(rest_client, db):
    response = rest_client.post(
        "/chat-api/v1/api/users/reset-password/",
        data={"email": "not_existing_email@example.com"},
        format="json",
    )
    assert response.status_code == 404
