import requests
import os

api_key = os.getenv("GOOGLE_MAPS_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY not set in environment")

url = "https://maps.googleapis.com/maps/api/staticmap"
params = {
    "center": "51.05,12.45",
    "zoom": 20,
    "size": "100x100",
    "maptype": "satellite",
    "key": api_key
}

r = requests.get(url, params=params)
print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('Content-Type', '')}")
print(f"Full Response:")
print(r.text)
