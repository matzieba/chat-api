def test_auth_view_has_access_to_user(client, firebase_accepts, user):
    response = client.get("/authenticated", HTTP_AUTHORIZATION="Bearer ValidToken")
    assert response.status_code == 200
    assert response.data == user["user_id"]


def test_auth_view_fails_on_no_header(client):
    response = client.get("/authenticated")
    assert response.status_code == 401


def test_auth_fails_on_firebase_reject(client, firebase_rejects):
    response = client.get("/authenticated", HTTP_AUTHORIZATION="Bearer InvalidToken")
    assert response.status_code == 401


def test_auth_fails_on_firebase_no_user(client, firebase_returns_no_user):
    response = client.get("/authenticated", HTTP_AUTHORIZATION="Bearer InvalidToken")
    assert response.status_code == 401
