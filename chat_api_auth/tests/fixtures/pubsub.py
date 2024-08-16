from unittest import mock

import pytest


@pytest.fixture()
def mock_google_oauth():
    with mock.patch("chat_api_auth.backends.pubsub.id_token", autospec=True) as mocked:
        yield mocked
