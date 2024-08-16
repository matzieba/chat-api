from drf_writable_nested import WritableNestedModelSerializer

from chat_api.clients.firebase import FirebaseClient
from chat_api.models import User
from chat_api.serializers.company import CompanySerializer


class UserSerializer(WritableNestedModelSerializer):
    class Meta:
        model = User
        read_only_fields = ["id", "firebase_uid"]
        fields = [
            "first_name",
            "last_name",
            "email",
            "phone",
            "id",
            "firebase_uid",
            "company",
            "job_title",
            "profile_picture",
        ]

    company = CompanySerializer(required=False, allow_null=True)

    def create(self, validated_data):
        if validated_data.get("email") and not validated_data.get("firebase_uid"):
            first_name = validated_data.get("first_name")
            last_name = validated_data.get("last_name")
            email = validated_data.get("email")
            validated_data["firebase_uid"] = FirebaseClient.create_firebase_user(
                first_name, last_name, email
            )
        return super().create(validated_data)
