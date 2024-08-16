import pytest


@pytest.fixture()
def default_user_loader():
    def user_loader(user_data):
        return user_data

    return user_loader


@pytest.fixture()
def pub_sub_user_loader():
    def user_loader():
        return {"username": "pubsub"}

    return user_loader
