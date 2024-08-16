from model_bakery import baker
import pytest

from chat_api.models import Company


@pytest.fixture()
def company():
    return baker.make(Company)


@pytest.fixture()
def company_2():
    return baker.make(Company)
