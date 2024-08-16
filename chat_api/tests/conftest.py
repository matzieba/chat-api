import pytest
from unittest import mock

from rest_framework.test import APIClient
from chat_api.clients.firebase import FirebaseClient
from chat_api.models import User

from .fixtures import *


@pytest.fixture
def user():
    return User(firebase_uid='123', email='user@example.com')


@pytest.fixture()
def rest_client():
    return APIClient()


@pytest.fixture()
def auth_client(rest_client):
    def auth_user(user):
        rest_client.force_authenticate(user)
        return rest_client

    return auth_user


@pytest.fixture(autouse=True)
def mock_firebase_admin():
    with mock.patch('initialize.initialize_app'):
        yield


@pytest.fixture()
def firebase_returned_id():
    return "firebase123456abcdef"


@pytest.fixture()
def mock_firebase_client_create_user_with_password_and_email(firebase_returned_id):
    with mock.patch.object(
        FirebaseClient, "create_firebase_user_with_username_and_password"
    ) as mocked_firebase_user:
        mocked_firebase_user.return_value = firebase_returned_id
        yield mocked_firebase_user


@pytest.fixture()
def mock_firebase_client_create_user(firebase_returned_id):
    with mock.patch.object(
        FirebaseClient, "create_firebase_user"
    ) as mocked_firebase_user:
        mocked_firebase_user.return_value = firebase_returned_id
        yield mocked_firebase_user


@pytest.fixture()
def mock_firebase_client_delete_user(firebase_returned_id):
    with mock.patch.object(
        FirebaseClient, "delete_firebase_user"
    ) as mocked_firebase_user:
        mocked_firebase_user.return_value = firebase_returned_id
        yield mocked_firebase_user
