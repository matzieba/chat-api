from unittest import mock

import pytest

from chat_api_auth.backends.firebase import FirebaseJWTBackend


@pytest.fixture()
def mock_firebase():
    with mock.patch(
        "chat_api_auth.backends.firebase.firebase_auth.verify_id_token", autospec=True
    ) as mocked_firebase:
        yield mocked_firebase


@pytest.fixture()
def firebase_accepts(mock_firebase, user):
    mock_firebase.return_value = user
    return mock_firebase


@pytest.fixture()
def firebase_returns_no_user(mock_firebase):
    mock_firebase.return_value = None
    return mock_firebase


@pytest.fixture()
def firebase_rejects(mock_firebase):
    mock_firebase.side_effect = [
        (ValueError("Token Expired")),
    ]
    return mock_firebase


@pytest.fixture()
def firebase_backend(default_user_loader):
    return FirebaseJWTBackend(default_user_loader)
