"""
Geocoding module - Convert addresses to coordinates using Mapbox Geocoding API
"""
import os
import requests
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Geocoder:
    """Converts addresses to latitude/longitude coordinates using Mapbox Geocoding API."""
    
    GEOCODING_URL = "https://api.mapbox.com/geocoding/v5/mapbox.places"
    
    def __init__(self, access_token: Optional[str] = None):
        """
        Initialize geocoder with Mapbox Geocoding API.
        
        Args:
            access_token: Mapbox access token (or set MAPBOX_ACCESS_TOKEN env var)
        """
        self.access_token = access_token or os.getenv("MAPBOX_ACCESS_TOKEN")
        if not self.access_token:
            raise ValueError(
                "Mapbox access token required for geocoding. "
                "Set MAPBOX_ACCESS_TOKEN environment variable."
            )
    
    def geocode(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Convert an address to latitude/longitude coordinates.
        
        Args:
            address: Street address, city, or location name
            
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        try:
            # URL encode the address
            import urllib.parse
            encoded_address = urllib.parse.quote(address)
            
            url = f"{self.GEOCODING_URL}/{encoded_address}.json"
            params = {
                "access_token": self.access_token,
                "limit": 1,
                "language": "de"  # German results preferred
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("features") and len(data["features"]) > 0:
                # Mapbox returns [longitude, latitude] order
                coords = data["features"][0]["geometry"]["coordinates"]
                return (coords[1], coords[0])  # Return as (lat, lon)
            else:
                print(f"Geocoding failed: No results found for '{address}'")
                return None
                
        except requests.RequestException as e:
            print(f"Geocoding error: {e}")
            return None
    
    def reverse_geocode(self, lat: float, lon: float) -> Optional[str]:
        """
        Convert coordinates back to an address.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Address string or None if not found
        """
        try:
            url = f"{self.GEOCODING_URL}/{lon},{lat}.json"
            params = {
                "access_token": self.access_token,
                "language": "de"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("features") and len(data["features"]) > 0:
                return data["features"][0]["place_name"]
            return None
                
        except requests.RequestException as e:
            print(f"Reverse geocoding error: {e}")
            return None


# Convenience function for quick usage
def get_coordinates(address: str) -> Optional[Tuple[float, float]]:
    """
    Quick function to get coordinates from an address.
    
    Args:
        address: Street address, city, or location name
        
    Returns:
        Tuple of (latitude, longitude) or None if not found
    """
    geocoder = Geocoder()
    return geocoder.geocode(address)


if __name__ == "__main__":
    # Test the geocoder
    test_address = "Alexanderplatz, Berlin, Germany"
    coords = get_coordinates(test_address)
    if coords:
        print(f"Address: {test_address}")
        print(f"Coordinates: {coords[0]:.6f}, {coords[1]:.6f}")
    else:
        print("Address not found")
