from chat_api_auth.django_auth.authentication import ChatApiDjangoAuthenticationToken
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters, generics, permissions, viewsets, status
from rest_framework.response import Response
from django.http import Http404

from chat_api.exceptions.user_does_not_exist import UserDoesNotExist
from chat_api.clients.firebase import FirebaseClient, FirebaseUserDeleteException
from chat_api.models import User
from chat_api.repositories.user import UserRepository
from chat_api.serializers.user import UserSerializer


class UserViewSetPermissions(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method == "DELETE":
            return request.user.company_id == obj.company_id
        if request.method in ["PUT", "PATCH"]:
            return request.user.firebase_uid == obj.firebase_uid
        return True


class UserViewSet(viewsets.ModelViewSet):
    authentication_classes = ()
    permission_classes = (UserViewSetPermissions,)
    serializer_class = UserSerializer
    queryset = User.objects.all()
    filter_backends = [DjangoFilterBackend, filters.SearchFilter]
    search_fields = ["first_name", "last_name", "job_title", "email"]

    def get_queryset(self):
        user = self.request.user
        if self.request.method in ["GET", "DELETE"]:
            return self.queryset.filter(company=user.company, company__isnull=False)
        return self.queryset


    def get_authenticators(self):
        if self.request.method == "POST":
            return ()
        return [
            ChatApiDjangoAuthenticationToken(),
        ]

    def destroy(self, request, *args, **kwargs):
        try:
            instance = self.get_object()
            FirebaseClient().delete_firebase_user(instance.firebase_uid)
            self.perform_destroy(instance)
            return Response(status=status.HTTP_204_NO_CONTENT)
        except FirebaseUserDeleteException as exc:
            return Response(status=status.HTTP_400_BAD_REQUEST, data=exc.args[0])



class UserMeView(generics.RetrieveAPIView):
    authentication_classes = (ChatApiDjangoAuthenticationToken,)
    permission_classes = ()
    serializer_class = UserSerializer

    def get_object(self):
        try:
            return UserRepository.get_me(self.request.user.firebase_uid)
        except UserDoesNotExist as e:
            raise Http404(e.args[0])


class UserLogoutAllSessionsView(generics.GenericAPIView):
    authentication_classes = (ChatApiDjangoAuthenticationToken,)
    permission_classes = ()
    serializer_class = ()

    def post(self, request, *args, **kwargs):
        FirebaseClient.logout_from_all_sessions(request.user.firebase_uid)
        return Response(status=201)
