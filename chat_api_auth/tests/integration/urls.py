from django.urls import path

from chat_api_auth.tests.integration.views import (
    AuthenticatedInternalView,
    AuthenticatedOptionalInternalAndTokenView,
    AuthenticatedOptionalView,
    AuthenticatedView,
    UnAuthenticatedView,
)

urlpatterns = [
    path("authenticated", AuthenticatedView.as_view()),
    path("optional_authenticated", AuthenticatedOptionalView.as_view()),
    path("un_authenticated", UnAuthenticatedView.as_view()),
    path("internal_authenticated", AuthenticatedInternalView.as_view()),
    path(
        "optional_authentication", AuthenticatedOptionalInternalAndTokenView.as_view()
    ),
]
