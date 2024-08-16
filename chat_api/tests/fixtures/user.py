from model_bakery import baker
import pytest

from chat_api.models import User


@pytest.fixture()
def user_data():
    return {
        "first_name": "John",
        "last_name": "Smith",
        "password": "password",
        "email": "user2@example.com",
        "status": "active",
        "firebase_uid": "b13321f23fgg1g23g24",
    }


@pytest.fixture()
def user2_data():
    return {
        "first_name": "Poor",
        "last_name": "Man",
        "password": "password",
        "email": "user6@example.com",
        "status": "active",
        "firebase_uid": "ajwcxhvh8a8",
    }


@pytest.fixture()
def user_with_company(company, user_data):
    return baker.make(
        User,
        email="user@app.io",
        firebase_uid=user_data["firebase_uid"],
        company=company,
    )


@pytest.fixture()
def user2_with_company(company_2, user2_data):
    return baker.make(User, firebase_uid=user2_data["firebase_uid"], company=company_2)
