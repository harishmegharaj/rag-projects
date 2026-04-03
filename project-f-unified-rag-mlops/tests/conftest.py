import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.api import app

    return TestClient(app)
