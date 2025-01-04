from unittest import mock

import pytest




def test_reset_password_for_existing_user(
    db, user_with_company, rest_client
):
    response = rest_client.post(
        "/chat-api/v1/api/users/reset-password/",
        data={"email": user_with_company.email},
        format="json",
    )

    assert response.status_code == 201


def test_reset_password_for_not_existing_user(rest_client, db):
    response = rest_client.post(
        "/chat-api/v1/api/users/reset-password/",
        data={"email": "not_existing_email@example.com"},
        format="json",
    )
    assert response.status_code == 404
