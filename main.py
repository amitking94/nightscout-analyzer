import os
import hashlib
import requests


NIGHTSCOUT_URL = os.environ["NIGHTSCOUT_URL"].rstrip("/")
API_SECRET = os.environ["NIGHTSCOUT_API_SECRET"]


def get_api_secret_hash(secret):
    return hashlib.sha1(secret.encode("utf-8")).hexdigest()


def test_connection():
    secret_hash = get_api_secret_hash(API_SECRET)

    headers = {
        "api-secret": secret_hash
    }

    url = f"{NIGHTSCOUT_URL}/api/v1/status.json"

    response = requests.get(url, headers=headers, timeout=20)

    print("HTTP status:", response.status_code)

    if response.status_code != 200:
        print("Nightscout response:")
        print(response.text)
        return False

    data = response.json()

    print("Nightscout connection successful!")
    print("Status:", data.get("status"))
    print("Version:", data.get("version"))

    return True


if __name__ == "__main__":
    test_connection()
