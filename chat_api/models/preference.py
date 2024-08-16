from django.db import models

from chat_api.models import User


class Preference(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    attending = models.BooleanField(null=True, blank=True)
    days_attending = models.IntegerField(null=True, blank=True)
    guest_number = models.IntegerField(null=True, blank=True)
    guest_names = models.JSONField(null=True, blank=True)
    needs_accommodation_help = models.BooleanField(null=True, blank=True)
    food_preference = models.CharField(max_length=255, null=True, blank=True)
    interested_in_top_of_babia_gora = models.BooleanField(null=True, blank=True)
    not_attending_reason = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.user.username