"""Geolocation Remediation Router.

Queries SerpApi Google Maps (engine="google_maps") to retrieve highly rated physical
pharmacies and surgical supply stores nearby when safety, regulatory, or pricing flags are triggered.

Filters strictly by rating >= 4.0 AND review_count > 10 to eliminate low-review anomalies.
Returns place_id and exact GPS coordinates to construct direct Google Maps turn-by-turn navigation URLs.
Labeling rule (Task T7 & T8): Results are labeled 'Highly rated nearby pharmacies', never 'verified'.
"""

import logging
import urllib.parse
from typing import List, Optional

import config
from schemas import NearbyPharmacy

logger = logging.getLogger(__name__)

# Curated reference stores strictly for offline demo replay
DEMO_REFERENCE_PHARMACIES = {
    "chennai": [
        NearbyPharmacy(
            name="SUKHAM HEALTHCARE [Demo Fixture]",
            rating=5.0,
            review_count=455,
            formatted_address="12th Main Rd, Anna Nagar West, Chennai, Tamil Nadu 600040",
            is_open=True,
            directions_url="https://www.google.com/maps/dir/?api=1&destination=13.0878,80.2085&destination_place_id=ChIJkUY7CxxkUjoR0xaCzDDYumU",
            place_id="ChIJkUY7CxxkUjoR0xaCzDDYumU",
            latitude=13.0878,
            longitude=80.2085,
            is_live_lookup=False
        ),
        NearbyPharmacy(
            name="Surgical Avenue [Demo Fixture]",
            rating=4.9,
            review_count=538,
            formatted_address="Mount Road, Chennai, Tamil Nadu 600002",
            is_open=True,
            directions_url="https://www.google.com/maps/dir/?api=1&destination=13.0645,80.2678&destination_place_id=ChIJ_8GKMeVlUjoR7J8ziOp9owQ",
            place_id="ChIJ_8GKMeVlUjoR7J8ziOp9owQ",
            latitude=13.0645,
            longitude=80.2678,
            is_live_lookup=False
        ),
    ]
}


def build_directions_url(name: str, place_id: Optional[str], lat: Optional[float], lng: Optional[float], city: str) -> str:
    """Construct precise Google Maps Directions URL using place_id or GPS coordinates."""
    if lat is not None and lng is not None:
        if place_id:
            return f"https://www.google.com/maps/dir/?api=1&destination={lat},{lng}&destination_place_id={place_id}"
        return f"https://www.google.com/maps/dir/?api=1&destination={lat},{lng}"
    
    encoded_dest = urllib.parse.quote_plus(f"{name} {city}")
    return f"https://www.google.com/maps/dir/?api=1&destination={encoded_dest}"


def find_nearby_verified_pharmacies(
    lat: float = config.DEFAULT_LAT,
    lng: float = config.DEFAULT_LNG,
    city_name: str = "Chennai",
    serpapi_key: Optional[str] = None,
    is_demo_mode: bool = False
) -> List[NearbyPharmacy]:
    """Retrieve highly rated nearby pharmacies with rating >= 4.0 and review_count > 10."""
    active_key = serpapi_key or config.SERPAPI_KEY

    # 1. Demo Mode
    if is_demo_mode:
        c_key = city_name.strip().lower()
        return DEMO_REFERENCE_PHARMACIES.get(c_key, DEMO_REFERENCE_PHARMACIES["chennai"])

    # 2. Live Mode WITHOUT API Key: Honest empty list
    if not active_key:
        logger.info("Live Google Maps lookup skipped: SERPAPI_KEY is not configured.")
        return []

    # 3. Live Mode WITH API Key: Real Google Maps query
    pharmacies: List[NearbyPharmacy] = []
    try:
        from serpapi import GoogleSearch
        print(f"[MAP ROUTER] Querying SerpApi Google Maps around ({lat}, {lng})...")

        params = {
            "engine": "google_maps",
            "q": "pharmacy medical surgical supply store",
            "ll": f"@{lat},{lng},14z",
            "hl": "en",
            "api_key": active_key
        }

        def _fetch_maps():
            return GoogleSearch(params).get_dict()

        results = config.execute_with_retry(
            _fetch_maps,
            timeout_seconds=15.0,
            max_retries=1,
            fallback={},
            label="Google Maps API"
        )
        local_results = results.get("local_results", []) if results else []

        for item in local_results:
            raw_rating = item.get("rating")
            raw_reviews = item.get("reviews")

            try:
                rating = float(raw_rating) if raw_rating is not None else None
            except ValueError:
                rating = None

            try:
                review_count = int(raw_reviews) if raw_reviews is not None else 0
            except ValueError:
                review_count = 0

            # Task T7: Filter by rating >= 4.0 AND review_count > 10
            if rating is None or rating < 4.0 or review_count <= 10:
                continue

            name = item.get("title", "Highly Rated Pharmacy")
            address = item.get("address", f"{city_name}, India")
            open_state = item.get("open_state")
            is_open = True if (open_state and "open" in open_state.lower()) else None
            place_id = item.get("place_id")
            
            gps = item.get("gps_coordinates") or {}
            item_lat = gps.get("latitude")
            item_lng = gps.get("longitude")

            directions_url = build_directions_url(name, place_id, item_lat, item_lng, city_name)

            pharmacies.append(
                NearbyPharmacy(
                    name=name,
                    rating=rating,
                    review_count=review_count,
                    formatted_address=address,
                    is_open=is_open,
                    directions_url=directions_url,
                    place_id=place_id,
                    latitude=item_lat,
                    longitude=item_lng,
                    is_live_lookup=True
                )
            )

        # Sort by review_count descending to prioritize highly established physical stores
        pharmacies.sort(key=lambda x: (x.review_count or 0, x.rating or 0.0), reverse=True)

    except Exception as e:
        logger.error(f"Live SerpApi Google Maps query failed: {e}")

    return pharmacies[:3]
