def test_warmup(rest_client):
    response = rest_client.get('/_ah/warmup')
    assert response.status_code == 200
