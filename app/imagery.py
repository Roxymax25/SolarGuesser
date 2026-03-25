"""
Imagery module - Fetch satellite images from Mapbox Static Images API
(Google Maps Static API satellite images are blocked for EEA accounts)
"""
import os
import requests
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SatelliteImagery:
    """Fetches satellite imagery from Mapbox Static Images API."""
    
    # Mapbox Static Images API base URL
    BASE_URL = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"
    
    # Resolution at different zoom levels (meters per pixel at equator)
    METERS_PER_PIXEL_ZOOM = {
        18: 0.597,
        19: 0.299,
        20: 0.149,
        21: 0.075,
    }
    
    def __init__(self, access_token: Optional[str] = None):
        """
        Initialize with Mapbox access token.
        
        Args:
            access_token: Mapbox access token (or set MAPBOX_ACCESS_TOKEN env var)
        """
        self.access_token = access_token or os.getenv("MAPBOX_ACCESS_TOKEN")
        if not self.access_token:
            raise ValueError(
                "Mapbox access token required. Set MAPBOX_ACCESS_TOKEN environment variable "
                "or pass access_token parameter. Get one free at https://mapbox.com"
            )
    
    def fetch_satellite_image(
        self,
        lat: float,
        lon: float,
        zoom: int = 18,
        size: Tuple[int, int] = (1280, 1280),
        highres: bool = True
    ) -> Optional[Image.Image]:
        """
        Fetch satellite image for given coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level (18-20 recommended for rooftop detail)
            size: Image size in pixels (max 1280x1280)
            highres: Use @2x for higher resolution
            
        Returns:
            PIL Image object or None if request failed
        """
        # Mapbox URL format: /lon,lat,zoom/widthxheight
        # Note: Mapbox uses lon,lat order (not lat,lon like Google)
        width, height = min(size[0], 1280), min(size[1], 1280)
        retina = "@2x" if highres else ""
        
        url = f"{self.BASE_URL}/{lon},{lat},{zoom}/{width}x{height}{retina}"
        params = {"access_token": self.access_token}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Check if we got an image
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                print(f"Error: Received non-image response: {content_type}")
                print(f"Response: {response.text[:500]}")
                return None
            
            image = Image.open(BytesIO(response.content))
            # Always convert to RGB (YOLO requires 3 channels)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
            
        except requests.RequestException as e:
            print(f"Error fetching satellite image: {e}")
            return None
    
    def get_meters_per_pixel(self, lat: float, zoom: int = 18) -> float:
        """
        Calculate meters per pixel at given latitude and zoom level.
        
        Args:
            lat: Latitude (resolution varies with latitude)
            zoom: Zoom level
            
        Returns:
            Meters per pixel
        """
        import math
        # Earth's circumference at equator in meters
        earth_circumference = 40075016.686
        # Pixels per tile
        pixels_per_tile = 256
        
        # Meters per pixel at equator for this zoom
        mpp_equator = earth_circumference / (pixels_per_tile * (2 ** zoom))
        
        # Adjust for latitude (cosine correction)
        # For @2x images, divide by 2
        mpp = mpp_equator * math.cos(math.radians(lat)) / 2
        
        return mpp


def fetch_satellite_image(
    lat: float,
    lon: float,
    zoom: int = 18
) -> Optional[Image.Image]:
    """
    Convenience function to fetch satellite image.
    
    Args:
        lat: Latitude
        lon: Longitude
        zoom: Zoom level (default 18 for good rooftop detail with Mapbox)
        
    Returns:
        PIL Image or None
    """
    imagery = SatelliteImagery()
    return imagery.fetch_satellite_image(lat, lon, zoom=zoom)


if __name__ == "__main__":
    # Test the imagery fetcher
    test_lat, test_lon = 52.5200, 13.4050  # Berlin
    
    imagery = SatelliteImagery()
    image = imagery.fetch_satellite_image(test_lat, test_lon, zoom=18)
    
    if image:
        print(f"Image fetched successfully: {image.size}")
        image.save("test_satellite.png")
        print("Saved as test_satellite.png")
    else:
        print("Failed to fetch image")
