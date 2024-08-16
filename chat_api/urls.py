from django.urls import include, path, re_path
from rest_framework.routers import DefaultRouter

from chat_api.views.conversation import ConversationViewSet
from chat_api.views.signup import SignupView, SignUpWithSSOView
from chat_api.views.user import UserLogoutAllSessionsView, UserMeView, UserViewSet
from chat_api.views.company import CompanyViewSet
from chat_api.views.reset_password import ResetPasswordView
from chat_api.views.user_invitation import UserInvitationViewSet

api_urls = [
    path(r"users/me", UserMeView.as_view(), name="me"),
    path(r"users/reset-password/", ResetPasswordView.as_view(), name="reset-password"),
    path(r"signup-with-sso/", SignUpWithSSOView.as_view(), name="signup-with-sso"),
    path(r"signup/", SignupView.as_view(), name="signup"),
    path(
        r"logout-all-sessions/",
        UserLogoutAllSessionsView.as_view(),
        name="logout-all-sessions",
    ),
    path(
    r"conversations/<int:pk>/messages/",
    ConversationViewSet.as_view({"post": "messages"}),
    name="conversations-messages",
),
]

default_router = DefaultRouter()

default_router.register(r"users", UserViewSet, basename="users")
default_router.register(r"companies", CompanyViewSet, basename="company")
default_router.register(
    r"user-invitations",
    UserInvitationViewSet,
    basename="invitation",
)

default_router.register(
    r"conversations",
    ConversationViewSet,
    basename="conversations",
)

api_urls += default_router.urls

urlpatterns = [
    re_path(r"api/", include(api_urls)),
]
