from rest_framework import serializers

from chat_api.models.preference import Preference


class PreferenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Preference
        exclude = ['user']