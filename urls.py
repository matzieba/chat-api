"""chat-api URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include, re_path

from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from chat_api.views.warmup import WarmUpView
from initialize import initialize_app

def extend_with_documentation(url_pattern, title, description):
    path = url_pattern.pattern
    schema_view = get_schema_view(
        openapi.Info(
            title=title,
            default_version="v1",
            description=description,
        ),
        patterns=[url_pattern],
        public=True,
        permission_classes=(permissions.AllowAny,),
        authentication_classes=(),
    )
    url_patterns = [
        url_pattern,
        re_path(
            f"{path}redoc/$",
            schema_view.with_ui("redoc", cache_timeout=0),
            name="schema-redoc",
        ),
        re_path(
            f"{path}swagger/$",
            schema_view.with_ui("swagger", cache_timeout=0),
            name="schema-swagger",
        ),
    ]
    return url_patterns

initialize_app()


urlpatterns = [
    path('admin/', admin.site.urls),
    re_path(r"^chat-api/v1/", include("chat_api.urls")),
    re_path(r"^_ah/warmup", WarmUpView.as_view(), name="warmup"),
]

urlpatterns.extend(
    extend_with_documentation(
        re_path(r"^chat-api/v1/", include("chat_api.urls")),
        title="chat_api",
        description=f"Behold My Awesome Project!",
    )
)