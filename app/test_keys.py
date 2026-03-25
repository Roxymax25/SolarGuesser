import requests
import os

api_key = os.getenv("GOOGLE_MAPS_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY not set in environment")

keys = [
    ("Google Maps key from env", api_key)
]

for name, key in keys:
    url = f"https://maps.googleapis.com/maps/api/staticmap?center=51.05,12.45&zoom=20&size=100x100&maptype=satellite&key={key}"
    r = requests.get(url)
    ct = r.headers.get('Content-Type', '')
    if 'image' in ct:
        print(f"{name}: ✅ OK (Status {r.status_code}, {ct})")
    else:
        print(f"{name}: ❌ FEHLER (Status {r.status_code})")
        print(f"   Response: {r.text[:100]}")
