from django.contrib.auth.models import AbstractUser
from django.db import models
from chat_api.clients.firebase import FirebaseClient

STATUS_CHOICES = [("active", "active"), ("disabled", "disabled")]


class User(AbstractUser):
    USERNAME_FIELD = "firebase_uid"
    REQUIRED_FIELDS = ["email"]

    username = None
    firebase_uid = models.CharField(unique=True, max_length=1024, null=True)
    first_name = models.CharField(max_length=1024)
    last_name = models.CharField(max_length=1024)
    phone = models.CharField(max_length=1024, null=True, blank=True)
    company = models.ForeignKey("Company", null=True, on_delete=models.SET_NULL)
    email = models.EmailField(max_length=1024, blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="active")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    job_title = models.CharField(max_length=1024, blank=True, null=True)
    profile_picture = models.ImageField(
        upload_to="user_profile_picture", null=True, blank=True
    )

    def __str__(self):
        return f"{self.first_name} {self.last_name or ''} - {self.id}"

    def delete(self, *args, **kwargs):
        FirebaseClient().delete_firebase_user(self.firebase_uid)
        super().delete(*args, **kwargs)
