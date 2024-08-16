from rest_framework import mixins, viewsets

from chat_api.models import Company
from chat_api.serializers.company import CompanySerializer
from chat_api_auth.django_auth.authentication import ChatApiDjangoAuthenticationToken


class CompanyViewSet(
    viewsets.GenericViewSet, mixins.UpdateModelMixin, mixins.RetrieveModelMixin
):
    authentication_classes = (ChatApiDjangoAuthenticationToken,)
    permission_classes = ()
    serializer_class = CompanySerializer

    def get_queryset(self):
        user = self.request.user
        return Company.objects.filter(pk=user.company.id)
