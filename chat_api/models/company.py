from django.db import models


class Company(models.Model):
    class Meta:
        verbose_name_plural = "Companies"

    name = models.CharField(max_length=1024)
    slug = models.CharField(max_length=1024)
    vat_nr = models.CharField(max_length=1024, null=True, blank=True)

    street = models.CharField(max_length=1024, blank=True, null=True)
    postal_code = models.CharField(max_length=1024, blank=True, null=True)
    city = models.CharField(max_length=1024, blank=True, null=True)
    country = models.CharField(max_length=1024, blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    primary_contact = models.ForeignKey(
        "User",
        null=True,
        on_delete=models.SET_NULL,
        related_name="primary_contact_for_companies",
    )
