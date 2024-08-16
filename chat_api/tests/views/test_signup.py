import copy

import pytest


@pytest.fixture()
def signup_data():
    return {
        "email": "test_email@cvt.services",
        "password": "adminadmin",
        "repeated_password": "adminadmin",
        "first_name": "John",
        "last_name": "Smith",
    }


def test_signup_view_correct_payload(
    rest_client,
    db,
    mock_firebase_client_create_user_with_password_and_email,
    firebase_returned_id,
    signup_data,
):
    response = rest_client.post(
        "/chat-api/v1/api/signup/", format="json", data=signup_data
    )
    assert response.status_code == 201


def test_signup_view_mismatched_passwords(
    rest_client,
    db,
    mock_firebase_client_create_user_with_password_and_email,
    signup_data,
):
    signup_data_mismatched_passwords = copy.deepcopy(signup_data)
    signup_data_mismatched_passwords.update(
        {"repeated_password": "randomwrongpassword"}
    )
    response = rest_client.post(
        "/chat-api/v1/api/signup/", format="json", data=signup_data_mismatched_passwords
    )
    assert response.status_code == 400
