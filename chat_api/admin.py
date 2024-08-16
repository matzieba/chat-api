from django.contrib import admin
from django import forms

from chat_api.clients.firebase import (
    FirebaseUserGetException,
    FirebaseUserUpdateException,
)
from chat_api.models import User
from chat_api.repositories.signup import SignupRepository
from chat_api.clients.firebase import FirebaseClient


# Register your models here.

class UserAdminForm(forms.ModelForm):

    password1 = forms.CharField(
        widget=forms.PasswordInput(),
        max_length=128,
        label="password",
        required=False,
        help_text="If you don't want to change the password leave fields empty.",
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput(),
        max_length=128,
        label="password (repeated)",
        required=False,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.instance.pk:
            self.fields["password1"].required = True
            self.fields["password2"].required = True
            self.fields["user_permissions"].disabled = True

    class Meta:
        model = User
        fields = (
            "first_name",
            "last_name",
            "company",
            "email",
            "status",
            "job_title",
            "profile_picture",
            "is_staff",
            "is_superuser",
            "firebase_uid",
            "password1",
            "password2",
            "user_permissions",
        )

    def save(self, commit=True):
        instance = super().save(commit=False)
        if self.cleaned_data.get("password1"):
            self.cleaned_data["password"] = self.cleaned_data["password1"]
        try:
            updated_keys = {
                k: v
                for k, v in self.cleaned_data.items()
                if k in ("email", "password") and v
            }
            if updated_keys:
                FirebaseClient().update_firebase_user(
                    firebase_uid=instance.firebase_uid,
                    **updated_keys,
                )
            if self.cleaned_data.get("password"):
                instance.set_password(self.cleaned_data["password"])
            instance.save()
        except (FirebaseUserGetException, FirebaseUserUpdateException):
            self.cleaned_data.pop("user_permissions", None)
            instance = SignupRepository().signup(self.cleaned_data)
            instance.set_password(self.cleaned_data["password"])
            instance.save()
        return instance


class UserAdmin(admin.ModelAdmin):
    list_display = (
        "full_name",
        "firebase_uid",
    )
    search_fields = ("full_name",)
    readonly_fields = ("firebase_uid",)

    form = UserAdminForm

    def full_name(self, obj):
        return f"{obj.first_name} {obj.last_name}"

    def save_model(self, request, obj, form, change):
        # entire save behaviour is changed in form already
        return


admin.site.register(User, UserAdmin)
