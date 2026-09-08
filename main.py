import os
import requests
from datetime import datetime, timedelta, timezone

NIGHTSCOUT_URL = os.environ["NIGHTSCOUT_URL"].rstrip("/")
ACCESS_TOKEN = os.environ["NIGHTSCOUT_ACCESS_TOKEN"]  # e.g. "readonly-app-xxxxxxxxxxxx"


def get_nightscout_data(endpoint, count=100):
    url = f"{NIGHTSCOUT_URL}/api/v1/{endpoint}"
    response = requests.get(
        url,
        params={"count": count, "token": ACCESS_TOKEN},
        timeout=30
    )
    print(f"{endpoint}: HTTP {response.status_code}")
    response.raise_for_status()
    return response.json()


def main():
    print("===================================")
    print(" Nightscout Data Collector")
    print("===================================")

    # Test status
    status = get_nightscout_data("status.json", 1)
    print("\nNightscout:")
    print("Status:", status.get("status"))
    print("Version:", status.get("version"))

    # Get recent glucose entries
    entries = get_nightscout_data("entries.json", 20)
    print("\nGlucose entries received:", len(entries))
    if entries:
        latest = entries[0]
        print("\nLatest glucose record:")
        print("Date:", latest.get("dateString"))
        print("SGV:", latest.get("sgv"))
        print("Direction:", latest.get("direction"))
        print("Device:", latest.get("device"))

    # Get recent treatments
    treatments = get_nightscout_data("treatments.json", 20)
    print("\nTreatment records received:", len(treatments))
    if treatments:
        print("\nRecent treatments:")
        for treatment in treatments[:5]:
            print(
                "-",
                treatment.get("eventType"),
                "| insulin:", treatment.get("insulin"),
                "| carbs:", treatment.get("carbs"),
                "| time:", treatment.get("created_at")
            )

    print("\n===================================")
    print(" Data collection successful")
    print("===================================")


if __name__ == "__main__":
    main()
