from django.http import HttpResponse
from rest_framework.routers import DefaultRouter
from django.urls import include, re_path, path
from chess_api.views import GameViewSet
def health(request):
    return HttpResponse("OK", status=200)
default_router = DefaultRouter()

default_router.register(r"chess_game", GameViewSet, basename="chess_game")

api_urls = []
api_urls += default_router.urls

urlpatterns = [
    re_path(r"api/", include(api_urls)),
    path('health', health),
]