import datetime
import uuid

from django.db import models

STATUS_CHOICES = [("pending", "pending"), ("accepted", "accepted")]


class UserInvitation(models.Model):
    USER_INVITATION_EXPIRY_LIMIT_DAYS = 1

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(max_length=1024, blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")

    company = models.ForeignKey("Company", on_delete=models.CASCADE)

    invited_by = models.ForeignKey(
        "User",
        null=True,
        on_delete=models.SET_NULL,
        related_name="invited_by",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def is_active(self):
        date_limit = datetime.datetime.now(self.created_at.tzinfo) - datetime.timedelta(
            days=self.USER_INVITATION_EXPIRY_LIMIT_DAYS
        )
        return self.status != "accepted" and self.created_at >= date_limit
