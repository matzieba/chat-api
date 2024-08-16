from rest_framework import serializers


class SignupSerializer(serializers.Serializer):
    password = serializers.CharField(max_length=1024)
    repeated_password = serializers.CharField(max_length=1024)
    first_name = serializers.CharField(max_length=1024)
    last_name = serializers.CharField(max_length=1024)
    company = serializers.CharField(max_length=1024, required=False, allow_blank=True)
    phone = serializers.CharField(max_length=1024, required=False, allow_blank=True)
    job_title = serializers.CharField(max_length=1024, required=False, allow_blank=True)
    email = serializers.CharField(max_length=1024)
    user_invitation_id = serializers.CharField(
        max_length=1024, required=False, allow_blank=True
    )

    def validate(self, data):
        if "password" and "repeated_password" in data:
            if len(data["password"]) < 6:
                raise serializers.ValidationError(
                    {"password": "Password length must be at least 6 characters"}
                )
            if data["password"] != data["repeated_password"]:
                raise serializers.ValidationError(
                    {"password": "Provided passwords don't match"}
                )

        if "user_invitation_id" in data:
            invitation = UserInvitationRepository.get_user_invitation(
                data["user_invitation_id"]
            )
            if data["email"] != invitation.email:
                raise serializers.ValidationError(
                    {"email": "You cannot change the invitation email address"}
                )

        return _validate_user_invitation(data)

class SSOSignupSerializer(serializers.Serializer):
    user_invitation_id = serializers.CharField(
        max_length=1024, required=False, allow_blank=True
    )

    def validate(self, data):
        return _validate_user_invitation(data)


def _validate_user_invitation(data: dict):
    if "user_invitation_id" in data and data["user_invitation_id"]:
        try:
            invitation = UserInvitationRepository.get_user_invitation(
                data["user_invitation_id"]
            )
        except UserInvitationDoesNotExistException:
            raise serializers.ValidationError({"invitation": "Invitation not found"})

        if not invitation.is_active:
            raise serializers.ValidationError(
                {"invitation": "Invitation is not active anymore"}
            )

        data["user_invitation"] = invitation

    return data