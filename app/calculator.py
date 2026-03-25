"""
Calculator module - Convert detected panel area to kWp estimation
"""
import math
from typing import Dict, Any, Optional


class KWpCalculator:
    """
    Calculates estimated kWp (kilowatt peak) from detected solar panel area.
    
    Standard Test Conditions (STC):
    - Irradiance: 1000 W/m²
    - Cell temperature: 25°C
    - Air mass: 1.5
    """
    
    # Typical solar panel efficiencies
    EFFICIENCY_LOW = 0.15      # Older panels
    EFFICIENCY_STANDARD = 0.18  # Standard modern panels
    EFFICIENCY_HIGH = 0.22     # Premium panels (e.g., SunPower, LG)
    
    # Typical panel specifications
    TYPICAL_PANEL_AREA_M2 = 1.7  # ~1m x 1.7m for standard residential panel
    TYPICAL_PANEL_POWER_W = 400   # Modern residential panel (400W)
    
    def __init__(
        self,
        panel_efficiency: float = 0.18,
        fill_factor: float = 0.85
    ):
        """
        Initialize the kWp calculator.
        
        Args:
            panel_efficiency: Solar cell efficiency (0.15-0.22 typical)
            fill_factor: Factor to account for gaps, frames, etc. in detection
                        (detected area vs actual cell area)
        """
        self.panel_efficiency = panel_efficiency
        self.fill_factor = fill_factor
    
    def pixels_to_square_meters(
        self,
        total_pixels: float,
        meters_per_pixel: float
    ) -> float:
        """
        Convert pixel area to square meters.
        
        Args:
            total_pixels: Total area in pixels (from detection)
            meters_per_pixel: Ground resolution (meters per pixel)
            
        Returns:
            Area in square meters
        """
        # Each pixel covers meters_per_pixel² square meters
        square_meters_per_pixel = meters_per_pixel ** 2
        return total_pixels * square_meters_per_pixel
    
    def calculate_kwp(
        self,
        area_m2: float,
        efficiency: Optional[float] = None
    ) -> float:
        """
        Calculate estimated kWp from panel area.
        
        Formula: kWp = Area (m²) × Efficiency × Fill Factor × 1 kW/m² (STC irradiance)
        
        Args:
            area_m2: Total solar panel area in square meters
            efficiency: Override default panel efficiency
            
        Returns:
            Estimated kWp
        """
        eff = efficiency or self.panel_efficiency
        
        # Apply fill factor to account for detection including frames, gaps
        effective_area = area_m2 * self.fill_factor
        
        # kWp = effective cell area × efficiency × 1 kW/m²
        kwp = effective_area * eff * 1.0
        
        return kwp
    
    def calculate_from_detection(
        self,
        detection_results: Dict[str, Any],
        meters_per_pixel: float,
        latitude: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate kWp from detection results.
        
        Args:
            detection_results: Results from SolarPanelDetector
            meters_per_pixel: Ground resolution
            latitude: Optional latitude for future tilt corrections
            
        Returns:
            Dictionary with area and power estimates
        """
        total_pixels = detection_results.get("total_panel_pixels", 0)
        num_panels = detection_results.get("num_panels", 0)
        
        # Convert pixels to square meters
        area_m2 = self.pixels_to_square_meters(total_pixels, meters_per_pixel)
        
        # Calculate kWp
        kwp = self.calculate_kwp(area_m2)
        
        # Estimate number of standard panels
        estimated_panels = area_m2 / self.TYPICAL_PANEL_AREA_M2
        
        # Calculate range (low to high efficiency)
        kwp_low = self.calculate_kwp(area_m2, self.EFFICIENCY_LOW)
        kwp_high = self.calculate_kwp(area_m2, self.EFFICIENCY_HIGH)
        
        return {
            "total_pixels": total_pixels,
            "area_m2": round(area_m2, 2),
            "kwp_estimate": round(kwp, 2),
            "kwp_range": {
                "low": round(kwp_low, 2),
                "high": round(kwp_high, 2)
            },
            "detected_panels": num_panels,
            "estimated_standard_panels": round(estimated_panels, 1),
            "meters_per_pixel": meters_per_pixel,
            "panel_efficiency_used": self.panel_efficiency,
            "annual_production_kwh_estimate": round(kwp * 950, 0)  # ~950 kWh/kWp in Germany
        }
    
    def estimate_annual_production(
        self,
        kwp: float,
        location: str = "germany"
    ) -> float:
        """
        Estimate annual electricity production.
        
        Args:
            kwp: Installed capacity in kWp
            location: Geographic location for irradiance factor
            
        Returns:
            Estimated annual production in kWh
        """
        # Typical specific yield (kWh per kWp per year)
        specific_yields = {
            "germany": 950,      # ~950 kWh/kWp
            "spain": 1500,       # ~1500 kWh/kWp
            "italy": 1300,       # ~1300 kWh/kWp
            "uk": 850,           # ~850 kWh/kWp
            "default": 1000
        }
        
        yield_factor = specific_yields.get(location.lower(), specific_yields["default"])
        return kwp * yield_factor


def calculate_kwp_from_pixels(
    total_pixels: float,
    meters_per_pixel: float,
    efficiency: float = 0.18
) -> Dict[str, float]:
    """
    Convenience function to calculate kWp from pixel area.
    
    Args:
        total_pixels: Detected panel area in pixels
        meters_per_pixel: Ground resolution
        efficiency: Panel efficiency
        
    Returns:
        Dictionary with area and kWp estimates
    """
    calculator = KWpCalculator(panel_efficiency=efficiency)
    area_m2 = calculator.pixels_to_square_meters(total_pixels, meters_per_pixel)
    kwp = calculator.calculate_kwp(area_m2)
    
    return {
        "area_m2": round(area_m2, 2),
        "kwp": round(kwp, 2)
    }


def get_meters_per_pixel(lat: float, zoom: int = 18, is_retina: bool = True) -> float:
    """
    Calculate ground resolution (meters per pixel) at given latitude and zoom.
    
    Args:
        lat: Latitude in degrees
        zoom: Map zoom level
        is_retina: If True, accounts for @2x retina images (Mapbox default)
        
    Returns:
        Meters per pixel
    """
    # Earth's circumference at equator in meters
    earth_circumference = 40075016.686
    # Standard tile size
    tile_size = 256
    
    # Meters per pixel at equator
    mpp_equator = earth_circumference / (tile_size * (2 ** zoom))
    
    # Adjust for latitude (cosine correction)
    mpp = mpp_equator * math.cos(math.radians(lat))
    
    # For @2x retina images, each pixel covers half the ground distance
    if is_retina:
        mpp = mpp / 2
    
    return mpp


if __name__ == "__main__":
    # Test the calculator
    print("kWp Calculator Test")
    print("=" * 40)
    
    calculator = KWpCalculator()
    
    # Example: 50,000 pixels detected at zoom 20, latitude 50°
    test_pixels = 50000
    test_mpp = get_meters_per_pixel(50, 20)
    
    print(f"Test parameters:")
    print(f"  Detected pixels: {test_pixels:,}")
    print(f"  Meters per pixel: {test_mpp:.4f}")
    
    result = calculate_kwp_from_pixels(test_pixels, test_mpp)
    
    print(f"\nResults:")
    print(f"  Area: {result['area_m2']} m²")
    print(f"  Estimated kWp: {result['kwp']} kWp")
    
    # Annual production estimate
    annual = calculator.estimate_annual_production(result['kwp'], "germany")
    print(f"  Annual production (Germany): ~{annual:.0f} kWh")
