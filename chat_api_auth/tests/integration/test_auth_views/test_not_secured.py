def test_un_auth_view(client):
    response = client.get("/un_authenticated")
    assert response.status_code == 200
