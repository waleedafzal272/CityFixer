"""
The City Fixer — Road Damage Geospatial Hotspot Clustering
==========================================================
Module for detecting high-density road damage hotspots using DBSCAN
(Density-Based Spatial Clustering of Applications with Noise)
over geographic GPS coordinates.

Features:
- Haversine distance metric for true spherical geographic distance.
- Configurable cluster radius (epsilon in meters) and density threshold (min_samples).
- Calculates aggregated cluster centers, hazard counts, and average severity index.
- Distinguishes between municipal priority clusters and isolated road damage.

Author : AI/ML Workspace
Module : Week 3 Surveillance & Spatial Analytics
"""

import math
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.cluster import DBSCAN

# Earth's mean radius in meters
EARTH_RADIUS_METERS = 6371000.0

# Severity weighting for numerical aggregation
SEVERITY_WEIGHTS = {
    "Tier 1": 3.0,
    "Tier 2": 2.0,
    "Tier 3": 1.0,
}

WEIGHT_TO_SEVERITY = {
    3.0: "Tier 1",
    2.0: "Tier 2",
    1.0: "Tier 3",
}


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two GPS points on Earth in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def cluster_detections(
    detections: List[Dict[str, Any]],
    eps_meters: float = 120.0,
    min_samples: int = 2,
) -> Dict[str, Any]:
    """
    Cluster road damage detections using DBSCAN with a Haversine distance metric.

    Parameters:
    -----------
    detections : list of dicts
        List of detection objects containing 'latitude' (or gps.latitude),
        'longitude' (or gps.longitude), 'severity', 'class_name', etc.
    eps_meters : float
        Maximum distance between two points to be considered in the same cluster.
        Default is 120 meters.
    min_samples : int
        Minimum number of samples required to form a high-density cluster hotspot.
        Default is 2 points.

    Returns:
    --------
    dict with:
        - total_detections: int
        - total_clusters: int
        - clustered_hazards_count: int
        - isolated_hazards_count: int
        - clusters: list of cluster summary dicts
        - noise: list of unclustered isolated detection points
    """
    if not detections:
        return {
            "total_detections": 0,
            "total_clusters": 0,
            "clustered_hazards_count": 0,
            "isolated_hazards_count": 0,
            "clusters": [],
            "noise": [],
        }

    # Normalize coordinate extraction
    cleaned_data = []
    coords_rad = []

    for d in detections:
        lat = d.get("latitude")
        lon = d.get("longitude")
        if lat is None or lon is None:
            gps = d.get("gps", {})
            lat = gps.get("latitude")
            lon = gps.get("longitude")

        if lat is None or lon is None:
            continue

        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            continue

        item = dict(d)
        item["latitude"] = lat
        item["longitude"] = lon
        cleaned_data.append(item)
        # DBSCAN haversine metric requires [lat, lon] in radians
        coords_rad.append([math.radians(lat), math.radians(lon)])

    if len(cleaned_data) < min_samples:
        return {
            "total_detections": len(cleaned_data),
            "total_clusters": 0,
            "clustered_hazards_count": 0,
            "isolated_hazards_count": len(cleaned_data),
            "clusters": [],
            "noise": cleaned_data,
        }

    # Convert distance in meters to radians for Haversine
    eps_radians = eps_meters / EARTH_RADIUS_METERS

    # Run DBSCAN
    coords_array = np.array(coords_rad)
    db = DBSCAN(eps=eps_radians, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(coords_array)

    # Group points by cluster label
    clusters_map: Dict[int, List[Dict[str, Any]]] = {}
    noise_points: List[Dict[str, Any]] = []

    for idx, label in enumerate(labels):
        point = cleaned_data[idx]
        if label == -1:
            noise_points.append(point)
        else:
            if label not in clusters_map:
                clusters_map[label] = []
            clusters_map[label].append(point)

    # Aggregate clusters
    clusters_summary = []
    total_clustered_pts = 0

    for cluster_id, pts in sorted(clusters_map.items()):
        total_clustered_pts += len(pts)
        lats = [p["latitude"] for p in pts]
        lons = [p["longitude"] for p in pts]

        center_lat = float(np.mean(lats))
        center_lon = float(np.mean(lons))

        # Calculate max radius from center in meters
        radius_meters = max(
            haversine_distance_meters(center_lat, center_lon, p["latitude"], p["longitude"])
            for p in pts
        )
        # Add slight buffer for visualization
        radius_meters = max(20.0, round(radius_meters, 1))

        # Severity analytics
        weights = [SEVERITY_WEIGHTS.get(p.get("severity", "Tier 3"), 1.0) for p in pts]
        avg_score = float(np.mean(weights))

        # Severity breakdown
        t1 = sum(1 for p in pts if p.get("severity") == "Tier 1")
        t2 = sum(1 for p in pts if p.get("severity") == "Tier 2")
        t3 = sum(1 for p in pts if p.get("severity") == "Tier 3")

        # Class breakdown
        classes: Dict[str, int] = {}
        for p in pts:
            cls = p.get("class_name", "unknown")
            classes[cls] = classes.get(cls, 0) + 1

        # Priority label
        if t1 > 0:
            dominant_severity = "Tier 1"
            priority = "CRITICAL (Immediate Maintenance)"
        elif t2 > 0 or avg_score >= 1.8:
            dominant_severity = "Tier 2"
            priority = "MODERATE (Scheduled Patching)"
        else:
            dominant_severity = "Tier 3"
            priority = "LOW (Monitoring Only)"

        clusters_summary.append({
            "cluster_id": int(cluster_id) + 1,
            "center": {
                "latitude": round(center_lat, 6),
                "longitude": round(center_lon, 6),
            },
            "radius_meters": radius_meters,
            "total_hazards": len(pts),
            "dominant_severity": dominant_severity,
            "average_severity_score": round(avg_score, 2),
            "priority": priority,
            "severity_breakdown": {
                "Tier 1": t1,
                "Tier 2": t2,
                "Tier 3": t3,
            },
            "class_breakdown": classes,
            "points": [
                {
                    "detection_id": p.get("detection_id"),
                    "class_name": p.get("class_name"),
                    "severity": p.get("severity"),
                    "confidence": p.get("confidence"),
                    "latitude": p.get("latitude"),
                    "longitude": p.get("longitude"),
                }
                for p in pts
            ],
        })

    # Sort clusters by total hazards & severity score descending
    clusters_summary.sort(
        key=lambda c: (c["severity_breakdown"]["Tier 1"], c["total_hazards"], c["average_severity_score"]),
        reverse=True,
    )

    return {
        "total_detections": len(cleaned_data),
        "total_clusters": len(clusters_summary),
        "clustered_hazards_count": total_clustered_pts,
        "isolated_hazards_count": len(noise_points),
        "clusters": clusters_summary,
        "noise": [
            {
                "detection_id": p.get("detection_id"),
                "class_name": p.get("class_name"),
                "severity": p.get("severity"),
                "latitude": p.get("latitude"),
                "longitude": p.get("longitude"),
            }
            for p in noise_points
        ],
    }


# Standalone CLI execution for testing
if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  🗺️ The City Fixer — DBSCAN Geospatial Clustering Test")
    print("=" * 60)

    # Sample GPS test points around GT Road Punjab
    sample_points = [
        {"detection_id": "D1", "latitude": 32.5731, "longitude": 74.0757, "severity": "Tier 1", "class_name": "pothole"},
        {"detection_id": "D2", "latitude": 32.5735, "longitude": 74.0761, "severity": "Tier 1", "class_name": "pothole"},
        {"detection_id": "D3", "latitude": 32.5732, "longitude": 74.0759, "severity": "Tier 2", "class_name": "alligator_crack"},
        # Second cluster 2km away
        {"detection_id": "D4", "latitude": 32.5920, "longitude": 74.0950, "severity": "Tier 2", "class_name": "transverse_crack"},
        {"detection_id": "D5", "latitude": 32.5923, "longitude": 74.0953, "severity": "Tier 3", "class_name": "crack"},
        # Isolated point
        {"detection_id": "D6", "latitude": 32.6500, "longitude": 74.2000, "severity": "Tier 3", "class_name": "pothole"},
    ]

    result = cluster_detections(sample_points, eps_meters=150.0, min_samples=2)
    print(f"Total Detections: {result['total_detections']}")
    print(f"Total Clusters  : {result['total_clusters']}")
    print(f"Clustered Points: {result['clustered_hazards_count']}")
    print(f"Noise (Isolated): {result['isolated_hazards_count']}")
    print("\nClusters Found:")
    for c in result["clusters"]:
        print(f"  • Cluster #{c['cluster_id']}: {c['total_hazards']} hazards at ({c['center']['latitude']}, {c['center']['longitude']}) | Priority: {c['priority']}")
    print("\nJSON Output Sample:")
    print(json.dumps(result["clusters"], indent=2))

