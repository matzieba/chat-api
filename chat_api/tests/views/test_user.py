import pytest
from model_bakery import baker
from rest_framework import status

from chat_api.models import Company, User


@pytest.fixture()
def user_create_data():
    return {
        "company": {"name": "CompanyName"},
        "first_name": "John",
        "last_name": "Smith",
        "password": "password",
        "email": "user@example.com",
        "status": "active",
    }


@pytest.fixture()
def user2_create_data():
    return {
        "first_name": "John",
        "last_name": "Smith",
        "password": "password",
        "email": "user2@example.com",
        "status": "active",
        "firebase_uid": "b13321f23fgg1g23g24",
    }


@pytest.fixture()
def company():
    return baker.make(Company)


@pytest.fixture()
def user_with_company(company, user):
    return baker.make(User, firebase_uid=user.firebase_uid, company=company)


def test_user_viewset(
    db,
    auth_client,
    rest_client,
    user,
    user_create_data,
    user2_create_data,
    firebase_returned_id,
    mock_firebase_client_create_user,
    mock_firebase_client_delete_user,
):
    response = rest_client.post(
        "/chat-api/v1/api/users/",
        format="json",
        data=user_create_data,
    )
    assert response.status_code == 201
    user = User.objects.filter(firebase_uid=firebase_returned_id).first()
    user.username = user.firebase_uid
    assert user
    assert Company.objects.filter(name=user_create_data["company"]["name"]).exists()

    assert user.first_name == user_create_data["first_name"]

    update_name_data = {"first_name": "Jim"}
    response = auth_client(user).patch(
        f"/chat-api/v1/api/users/{user.id}/",
        format="json",
        data=update_name_data,
    )

    assert response.status_code == 200
    user.refresh_from_db()
    assert user.first_name != user_create_data["first_name"]
    assert user.first_name == update_name_data["first_name"]

    response = auth_client(user).get(
        f"/chat-api/v1/api/users/",
        format="json",
    )
    assert response.status_code == 200
    assert len(response.data["results"]) == User.objects.count()

    user2 = User(**user2_create_data)
    user2.save()

    response = auth_client(user).delete(
        f"/chat-api/v1/api/users/{user2.id}/",
        format="json",
    )
    assert response.status_code == 404

    response = auth_client(user).delete(
        f"/chat-api/v1/api/users/{user.id}/",
        format="json",
    )
    assert response.status_code == 204
    assert not User.objects.filter(firebase_uid=firebase_returned_id).first()

    response = auth_client(user).delete(
        f"/chat-api/v1/api/users/321/",
        format="json",
    )
    assert response.status_code == 404
    assert not User.objects.filter(firebase_uid=firebase_returned_id).first()


def test_user_me(db, auth_client, user, user_with_company):
    response = auth_client(user).get(
        f"/chat-api/v1/api/users/me",
        format="json",
    )
    company = user_with_company.company
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "id": user_with_company.id,
        "company": {
            "id": company.id,
            "name": company.name,
        },
        "firebase_uid": user_with_company.firebase_uid,
        "first_name": user_with_company.first_name,
        "last_name": user_with_company.last_name,
        "phone": user_with_company.phone,
        "email": user_with_company.email,
        "job_title": None,
        "profile_picture": None,
    }


def test_user_me_not_authenticated_user(db, rest_client):
    response = rest_client.get(
        f"/chat-api/v1/api/users/me/",
        format="json",
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
