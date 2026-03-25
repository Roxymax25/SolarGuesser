"""
Solar Guesser - Main Streamlit Application

Web interface for estimating solar panel capacity from satellite imagery.
"""
import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import cv2

# Import our modules
from geocoder import Geocoder
from imagery import SatelliteImagery
from detector import SolarPanelDetector
from calculator import KWpCalculator, get_meters_per_pixel


# Page configuration
st.set_page_config(
    page_title="Solar Guesser",
    page_icon="☀️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B00;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B00;
    }
    .big-number {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B00;
    }
</style>
""", unsafe_allow_html=True)


# Cache the detector to avoid reloading on every interaction
@st.cache_resource
def load_detector():
    """Load the specialized solar panel detector (cached)."""
    return SolarPanelDetector(
        confidence_threshold=0.25
    )


@st.cache_resource
def load_imagery():
    """Load the imagery fetcher (cached)."""
    return SatelliteImagery()


@st.cache_resource
def load_geocoder():
    """Load the geocoder (cached)."""
    return Geocoder()


def main():
    # Header
    st.markdown('<p class="main-header">☀️ Solar Guesser</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Schätze die installierte Solarleistung auf Dächern</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar settings
    st.sidebar.header("⚙️ Einstellungen")
    
    zoom_level = st.sidebar.slider(
        "Zoom-Level",
        min_value=16,
        max_value=20,
        value=18,
        help="Höherer Zoom = mehr Details, aber kleinerer Bereich (Mapbox: 18 optimal)"
    )
    
    confidence = st.sidebar.slider(
        "Erkennungs-Konfidenz",
        min_value=0.01,
        max_value=0.9,
        value=0.1,
        step=0.01,
        help="Niedrigere Werte = mehr Erkennungen, aber auch mehr Falsch-Positive"
    )
    
    panel_efficiency = st.sidebar.select_slider(
        "Panel-Effizienz",
        options=[0.15, 0.18, 0.20, 0.22],
        value=0.18,
        format_func=lambda x: f"{int(x*100)}%",
        help="Typische Effizienz: 15% (alt) bis 22% (Premium)"
    )
    
    # Main content - two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📍 Eingabe")
        
        input_method = st.radio(
            "Wähle eine Methode:",
            ["Adresse eingeben", "Bild hochladen"],
            horizontal=True
        )
        
        image = None
        latitude = 50.0  # Default for Germany
        
        if input_method == "Adresse eingeben":
            address = st.text_input(
                "Adresse",
                placeholder="z.B. Hauptstraße 1, 12345 Berlin",
                help="Gib eine vollständige Adresse ein"
            )
            
            if st.button("🔍 Satellitenimage abrufen", type="primary"):
                if address:
                    with st.spinner("Suche Koordinaten..."):
                        geocoder = load_geocoder()
                        coords = geocoder.geocode(address)
                        
                        if coords:
                            latitude, longitude = coords
                            st.success(f"📍 Gefunden: {latitude:.5f}, {longitude:.5f}")
                            
                            with st.spinner("Lade Satellitenimage..."):
                                imagery = load_imagery()
                                image = imagery.fetch_satellite_image(
                                    latitude, longitude,
                                    zoom=zoom_level
                                )
                                
                                if image:
                                    # Ensure RGB format for YOLO
                                    if image.mode != 'RGB':
                                        image = image.convert('RGB')
                                    st.session_state['image'] = image
                                    st.session_state['latitude'] = latitude
                                    st.session_state['zoom'] = zoom_level
                                else:
                                    st.error("Fehler beim Laden des Satellitenbildes")
                        else:
                            st.error("Adresse nicht gefunden. Bitte überprüfe die Eingabe.")
                else:
                    st.warning("Bitte gib eine Adresse ein")
        
        else:  # Bild hochladen
            uploaded_file = st.file_uploader(
                "Satellitenbild hochladen",
                type=["png", "jpg", "jpeg"],
                help="Lade ein Satellitenbild (z.B. von Google Earth) hoch"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                # Ensure RGB format for YOLO
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                st.session_state['image'] = image
                st.session_state['latitude'] = latitude
                st.session_state['zoom'] = zoom_level
                
                # Ask for scale if uploading custom image
                st.info("Bei hochgeladenen Bildern wird die Standardauflösung für Zoom 20 verwendet.")
        
        # Show the image with ROI selection if we have one
        if 'image' in st.session_state and st.session_state['image'] is not None:
            st.markdown("---")
            st.markdown("### ✏️ Bereich auswählen (optional)")
            
            # Get image dimensions
            img = st.session_state['image']
            img_width, img_height = img.size
            
            # ROI toggle
            use_roi = st.checkbox("🎯 ROI aktivieren (Bereich eingrenzen)", key="use_roi")
            
            if use_roi:
                st.caption("Wähle den Bildbereich, der analysiert werden soll:")
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    x_min = st.slider("Links (X Start)", 0, img_width - 10, 0, key="roi_x_min")
                    y_min = st.slider("Oben (Y Start)", 0, img_height - 10, 0, key="roi_y_min")
                
                with col_right:
                    x_max = st.slider("Rechts (X Ende)", x_min + 10, img_width, img_width, key="roi_x_max")
                    y_max = st.slider("Unten (Y Ende)", y_min + 10, img_height, img_height, key="roi_y_max")
                
                # Store ROI coordinates
                st.session_state['roi_coords'] = (x_min, y_min, x_max, y_max)
                
                # Show preview with ROI overlay
                preview_array = np.array(img).copy()
                # Dim outside ROI
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                mask[y_min:y_max, x_min:x_max] = 255
                dimmed = (preview_array * 0.3).astype(np.uint8)
                preview_array[mask == 0] = dimmed[mask == 0]
                # Draw rectangle
                cv2.rectangle(preview_array, (x_min, y_min), (x_max, y_max), (255, 107, 0), 3)
                
                st.image(Image.fromarray(preview_array), caption=f"ROI: {x_max - x_min} x {y_max - y_min} px", use_container_width=True)
                
                roi_area = (x_max - x_min) * (y_max - y_min)
                total_area = img_width * img_height
                st.caption(f"📐 Ausgewählter Bereich: {roi_area:,} px² ({roi_area/total_area*100:.1f}% des Bildes)")
            else:
                # No ROI, show normal image
                if 'roi_coords' in st.session_state:
                    del st.session_state['roi_coords']
                st.image(img, caption="Satellitenbild", use_container_width=True)
    
    with col2:
        st.subheader("📊 Analyse")
        
        if 'image' in st.session_state and st.session_state['image'] is not None:
            # Show ROI status
            has_roi = 'roi_coords' in st.session_state
            if has_roi:
                st.info("🎯 ROI aktiv: Nur der markierte Bereich wird analysiert")
            
            if st.button("🔬 Solarpanele erkennen", type="primary"):
                with st.spinner("Lade KI-Modell und analysiere Bild..."):
                    try:
                        # Load detector
                        detector = load_detector()
                        detector.confidence_threshold = confidence
                        
                        original_image = st.session_state['image']
                        img_width, img_height = original_image.size
                        
                        # Prepare image for detection (crop to ROI if defined)
                        if has_roi:
                            x_min, y_min, x_max, y_max = st.session_state['roi_coords']
                            
                            # Crop the image to ROI
                            cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
                            
                            st.caption(f"📐 ROI-Bereich: {x_max - x_min} x {y_max - y_min} px")
                            
                            # Run detection on cropped image
                            results = detector.detect(cropped_image)
                            
                            # Create annotated image showing ROI on original
                            annotated_cropped = np.array(results['annotated_image'])
                            
                            # Build full annotated image with ROI overlay
                            original_array = np.array(original_image)
                            # Dim areas outside ROI
                            dimmed = (original_array * 0.3).astype(np.uint8)
                            full_annotated = dimmed.copy()
                            
                            # Paste the annotated cropped section back
                            full_annotated[y_min:y_max, x_min:x_max] = annotated_cropped
                            
                            # Draw ROI border
                            cv2.rectangle(full_annotated, (x_min, y_min), (x_max, y_max), (255, 107, 0), 3)
                            
                            results['annotated_image'] = Image.fromarray(full_annotated)
                            results['crop_offset'] = (x_min, y_min)
                            
                            st.success(f"🎯 {results['num_panels']} Panel(s) im ROI erkannt")
                        else:
                            # No ROI, detect on full image
                            results = detector.detect(original_image)
                        
                        # Calculate kWp
                        lat = st.session_state.get('latitude', 50.0)
                        zoom = st.session_state.get('zoom', 18)
                        mpp = get_meters_per_pixel(lat, zoom, is_retina=True)
                        
                        # Debug info
                        st.caption(f"🔍 Debug: Zoom={zoom}, mpp={mpp:.4f} m/px, Total pixels={results['total_panel_pixels']:,}")
                        
                        calculator = KWpCalculator(panel_efficiency=panel_efficiency)
                        power_results = calculator.calculate_from_detection(results, mpp)
                        
                        # Store results in session
                        st.session_state['results'] = results
                        st.session_state['power_results'] = power_results
                        
                    except Exception as e:
                        st.error(f"Fehler bei der Analyse: {e}")
                        st.exception(e)
        
        # Show results if available
        if 'results' in st.session_state and st.session_state['results'] is not None:
            results = st.session_state['results']
            power_results = st.session_state['power_results']
            
            # Annotated image
            st.image(
                results['annotated_image'],
                caption=f"Erkannte Solarpanele: {results['num_panels']}",
                width="stretch"
            )
            
            # Results display
            st.markdown("---")
            
            # Main result
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    label="Geschätzte Leistung",
                    value=f"{power_results['kwp_estimate']} kWp",
                    delta=None
                )
            
            with col_b:
                st.metric(
                    label="Panelfläche",
                    value=f"{power_results['area_m2']} m²",
                    delta=None
                )
            
            with col_c:
                st.metric(
                    label="Erkannte Panels",
                    value=results['num_panels'],
                    delta=None
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional info
            st.markdown("---")
            
            with st.expander("📈 Detaillierte Ergebnisse"):
                st.write(f"**Leistungsbereich:** {power_results['kwp_range']['low']} - {power_results['kwp_range']['high']} kWp")
                st.write(f"**Geschätzte Standardpanels:** ~{power_results['estimated_standard_panels']} Stück")
                st.write(f"**Jährliche Produktion (DE):** ~{power_results['annual_production_kwh_estimate']:,.0f} kWh")
                st.write(f"**Verwendete Effizienz:** {power_results['panel_efficiency_used']*100:.0f}%")
                st.write(f"**Auflösung:** {power_results['meters_per_pixel']:.4f} m/Pixel")
            
            with st.expander("ℹ️ Hinweise zur Genauigkeit"):
                st.markdown("""
                - ✅ Die Schätzung basiert auf **präziser Segmentierung** (nicht nur Bounding Boxes)
                - Dachneigung wird aktuell nicht berücksichtigt (projizierte Fläche)
                - Tatsächliche kWp kann je nach Paneltyp und -alter variieren
                - Bei schlechter Bildqualität kann die Erkennung ungenau sein
                - **Modell:** YOLO26m-seg mit mAP50 = 41%
                """)
        
        elif 'image' not in st.session_state:
            st.info("👈 Gib eine Adresse ein oder lade ein Bild hoch, um zu starten")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>Solar Guesser v1.1 | "
        "Powered by YOLO26m-seg & Google Maps</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
