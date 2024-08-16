import base64


def test_optional_auth_view_200_on_basic_auth(client):
    response = client.get(
        "/optional_authentication",
        HTTP_AUTHORIZATION="Basic "
        + base64.b64encode(b"test_service:test_service_password").decode("utf-8"),
        HTTP_INTERNAL_AUTH=True,
    )
    assert response.status_code == 200


def test_optional_auth_view_200_on_token_access(client, firebase_accepts, user):
    response = client.get(
        "/optional_authentication", HTTP_AUTHORIZATION="Bearer ValidToken"
    )
    assert response.status_code == 200


def test_403_on_no_auth(client):
    response = client.get("/optional_authentication")
    assert response.status_code == 403
