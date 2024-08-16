import pytest


# Example user which will be extracted from firebase JWT
@pytest.fixture()
def user():
    return {
        "name": "Jakub Jakubowski",
        "picture": "https://lh6.googleusercontent.com/-ztDj-NVlEt0/AAAAAAAAAAI/AAAAAAAAAAc/O9JVb7JfbOo/photo.jpg",
        "iss": "https://securetoken.google.com/mageekz-2018",
        "aud": "mageekz-2018",
        "auth_time": 1552493659,
        "user_id": "g5enViJK6MhW2J8mi7BzMXdwnZM2",
        "sub": "g5enViJK6MhW2J8mi7BzMXdwnZM2",
        "iat": 1552493659,
        "exp": 1552497259,
        "email": "jk@tb4hr.com",
        "email_verified": True,
        "firebase": {
            "identities": {
                "google.com": ["108042251392836880173"],
                "email": ["jk@tb4hr.com"],
            },
            "sign_in_provider": "google.com",
        },
        "uid": "g5enViJK6MhW2J8mi7BzMXdwnZM2",
    }


@pytest.fixture()
def internal_user():
    return {"service_name": "test_service"}
