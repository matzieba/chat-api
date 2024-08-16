import base64


def test_auth_task_view_200_on_correct_task_queue_name_in_header(client):
    response = client.get(
        "/internal_authenticated",
        HTTP_AUTHORIZATION="Basic "
        + base64.b64encode(b"test_service:test_service_password").decode("utf-8"),
        HTTP_INTERNAL_AUTH=True,
    )
    assert response.status_code == 200


def test_auth_task_view_fails_on_header(client):
    response = client.get("/internal_authenticated", HTTP_INTERNAL_AUTH=True)
    assert response.status_code == 401


def test_auth_task_view_fails_on_invalid_username(client):
    response = client.get(
        "/internal_authenticated",
        HTTP_AUTHORIZATION="Basic " + base64.b64encode(b"uname124:pw").decode("utf-8"),
        HTTP_INTERNAL_AUTH=True,
    )
    assert response.status_code == 401


def test_auth_task_view_fails_on_invalid_password(client):
    response = client.get(
        "/internal_authenticated",
        HTTP_AUTHORIZATION="Basic "
        + base64.b64encode(b"test_service:pw").decode("utf-8"),
        HTTP_INTERNAL_AUTH=True,
    )
    assert response.status_code == 401
