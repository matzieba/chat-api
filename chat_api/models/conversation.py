from django.db import models



class Conversation(models.Model):
    user = models.ForeignKey(
        "User", on_delete=models.PROTECT, related_name="conversations"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    preferences_gathered = models.BooleanField(default=False)

    def __str__(self):
        return f"Conversation for {self.user.username} with type {self.type}"
