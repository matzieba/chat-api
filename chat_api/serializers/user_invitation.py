from django.core.exceptions import ObjectDoesNotExist
from rest_framework import serializers
from rest_framework.serializers import ModelSerializer

from chat_api.models import User
from chat_api.models.user_invitation import UserInvitation
from chat_api.serializers.company import CompanySerializer


class CreateUserInvitationSerializer(serializers.Serializer):
    email = serializers.CharField(max_length=1024)

    def validate(self, data):
        try:
            User.objects.get(
                email=data["email"], company=self.context["request"].user.company
            )
        except ObjectDoesNotExist:
            return data

        raise serializers.ValidationError({"user": "User already exists"})


class UserInvitationSerializer(ModelSerializer):
    company = CompanySerializer(read_only=True)

    class Meta:
        model = UserInvitation
        fields = [
            "id",
            "email",
            "company",
        ]
