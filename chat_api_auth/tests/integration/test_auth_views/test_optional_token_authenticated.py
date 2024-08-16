def test_auth_view_success_no_user(client):
    response = client.get("/optional_authenticated")
    assert response.data is None
    assert response.status_code == 200


def test_auth_view_success_user(client, firebase_accepts, user):
    response = client.get(
        "/optional_authenticated", HTTP_AUTHORIZATION="Bearer ValidToken"
    )
    assert response.data == user["user_id"]
    assert response.status_code == 200
