import pytest


@pytest.fixture(autouse=True)
def use_dummy_app(settings, db):
    settings.INSTALLED_APPS.append("chat_api_auth.tests.integration")
    overwrites = dict(
        SECRET_KEY="not very secret in tests",
        ROOT_URLCONF="chat_api_auth.tests.integration.urls",
        REST_FRAMEWORK={
            "DEFAULT_PERMISSION_CLASSES": (
                "rest_framework.permissions.IsAuthenticated",
            ),
        },
        AUTH={
            "internal_services": {
                "test_service": "test_service_password",
            },
            "pub_sub_project_id": "test-api",
        },
    )

    for key, value in overwrites.items():
        setattr(settings, key, value)
