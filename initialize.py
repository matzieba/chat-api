import firebase_admin
from firebase_admin import credentials

from django.conf import settings


def initialize_app():
    try:
        cred = credentials.Certificate(settings.FIREBASE_CERT)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        print(e)
