import pytest

from chat_api_auth.backends.internal import InternalBackend


@pytest.fixture()
def internal_backend(default_user_loader):
    return InternalBackend(default_user_loader)
