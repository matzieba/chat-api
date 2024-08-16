from rest_framework import generics
from rest_framework.response import Response


class WarmUpView(generics.GenericAPIView):
    authentication_classes = ()
    permission_classes = ()

    def get(self, request, *args, **kwargs):
        return Response(status=200)
