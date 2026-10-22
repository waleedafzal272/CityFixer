"""
The City Fixer — Municipal Road Health & Maintenance Brief (PDF Generator)
==========================================================================
Generates a formal, high-resolution PDF document summarizing road degradation,
damage detection counts, DBSCAN high-density hotspot clusters, and road quality index
for municipal engineers, city commissioners, and maintenance crews.

Author : AI/ML Workspace
Module : Week 4 Stretch Analytics & Reporting
"""

import os
import io
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

# DBSCAN clustering integration
try:
    from backend.clustering import cluster_detections
except ImportError:
    from clustering import cluster_detections

# ReportLab imports
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        KeepTogether,
        HRFlowable,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "city_fixer.db")


def query_system_metrics(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Fetch all detections and calculate summary statistics."""
    if not os.path.exists(db_path):
        return {
            "total": 0,
            "tier1": 0,
            "tier2": 0,
            "tier3": 0,
            "health_index": 100,
            "avg_conf": 0.0,
            "classes": {},
            "detections": [],
            "clusters": [],
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM detections ORDER BY timestamp DESC").fetchall()
    conn.close()

    total = len(rows)
    tier1 = sum(1 for r in rows if r["severity"] == "Tier 1")
    tier2 = sum(1 for r in rows if r["severity"] == "Tier 2")
    tier3 = sum(1 for r in rows if r["severity"] == "Tier 3")

    # Weighted penalty for road health index
    penalty = (tier1 * 3.5) + (tier2 * 1.8) + (tier3 * 0.6)
    health_index = max(10, round(100 - penalty)) if total > 0 else 100

    avg_conf = (
        round(sum(r["confidence"] for r in rows) / total, 4) if total > 0 else 0.0
    )

    classes: Dict[str, int] = {}
    detections: List[Dict[str, Any]] = []
    for r in rows:
        cls = r["class_name"]
        classes[cls] = classes.get(cls, 0) + 1
        detections.append({
            "detection_id": r["detection_id"],
            "class_name": r["class_name"],
            "confidence": r["confidence"],
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "severity": r["severity"],
            "timestamp": r["timestamp"],
        })

    # Run DBSCAN hotspot clustering
    cluster_res = cluster_detections(detections, eps_meters=150.0, min_samples=2)

    return {
        "total": total,
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "health_index": health_index,
        "avg_conf": avg_conf,
        "classes": classes,
        "detections": detections,
        "clusters": cluster_res.get("clusters", []),
        "noise_count": cluster_res.get("isolated_hazards_count", 0),
    }


def generate_pdf_report(
    db_path: str = DEFAULT_DB_PATH,
    output_path: Optional[str] = None,
) -> bytes:
    """
    Generate the official 'Municipal Road Health & Maintenance Brief' as a PDF.
    Returns binary PDF bytes and optionally writes to output_path.
    """
    metrics = query_system_metrics(db_path)
    buffer = io.BytesIO()

    if not HAS_REPORTLAB:
        # Fallback minimal text PDF if reportlab is not present
        pdf_content = (
            f"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            f"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            f"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<<>>>>endobj\n"
            f"4 0 obj<</Length 150>>stream\nBT /F1 18 Tf 50 720 Td (The City Fixer - Road Health Brief) Tj "
            f"/F1 12 Tf 0 -30 Td (Total Hazards: {metrics['total']} | Health Index: {metrics['health_index']}%) Tj ET\nendstream\nendobj\n"
            f"xref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000059 00000 n\n0000000115 00000 n\n0000000215 00000 n\n"
            f"trailer<</Size 5/Root 1 0 R>>\nstartxref\n415\n%%EOF"
        ).encode("latin-1")
        if output_path:
            with open(output_path, "wb") as f:
                f.write(pdf_content)
        return pdf_content

    # Initialize Document with standard margins
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom Design Palette
    c_primary = colors.HexColor("#0f172a")      # Slate 900
    c_navy = colors.HexColor("#1e293b")         # Slate 800
    c_blue = colors.HexColor("#2563eb")         # Blue 600
    c_rose = colors.HexColor("#e11d48")         # Rose 600
    c_amber = colors.HexColor("#d97706")        # Amber 600
    c_emerald = colors.HexColor("#059669")      # Emerald 600
    c_bg_light = colors.HexColor("#f8fafc")     # Slate 50
    c_border = colors.HexColor("#cbd5e1")       # Slate 300

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=c_primary,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
    )
    section_h1 = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=c_primary,
        spaceAfter=6,
    )
    body_text = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    cell_bold = ParagraphStyle(
        "CellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=c_primary,
    )
    cell_text = ParagraphStyle(
        "CellText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#334155"),
    )

    story = []

    # 1. Header Block with Municipal Emblem / Badge
    header_data = [
        [
            Paragraph("<b>MUNICIPAL INFRASTRUCTURE COMMISSION</b><br/><font color='#64748b' size='8'>Department of Urban Roadways & Highway Maintenance</font>", subtitle_style),
            Paragraph(f"<b>CONFIDENTIAL BRIEF</b><br/><font color='#64748b' size='8'>Doc ID: CF-MB-{datetime.now(timezone.utc).strftime('%Y%m%d')}</font>", subtitle_style),
        ]
    ]
    t_header = Table(header_data, colWidths=[340, 200])
    t_header.setStyle(TableStyle([
        ("ALIGN", (0, 0), (0, 0), "LEFT"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_blue, spaceBefore=0, spaceAfter=12))

    # 2. Document Title
    story.append(Paragraph("Municipal Road Health & Maintenance Brief", title_style))
    story.append(Spacer(1, 3))
    now_str = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    story.append(Paragraph(f"Autonomous Surveillance Telemetry & Pothole Hotspot Report &bull; Generated {now_str}", subtitle_style))
    story.append(Spacer(1, 14))

    # 3. KPI Metric Cards Table
    kpi_data = [
        [
            Paragraph("<font color='#64748b' size='7.5'>TOTAL DETECTIONS</font><br/><b><font size='18' color='#0f172a'>" + str(metrics["total"]) + "</font></b>", cell_bold),
            Paragraph("<font color='#e11d48' size='7.5'>TIER 1 (CRITICAL)</font><br/><b><font size='18' color='#e11d48'>" + str(metrics["tier1"]) + "</font></b>", cell_bold),
            Paragraph("<font color='#d97706' size='7.5'>TIER 2 (MODERATE)</font><br/><b><font size='18' color='#d97706'>" + str(metrics["tier2"]) + "</font></b>", cell_bold),
            Paragraph("<font color='#059669' size='7.5'>ROAD HEALTH INDEX</font><br/><b><font size='18' color='#059669'>" + str(metrics["health_index"]) + "%</font></b>", cell_bold),
        ]
    ]
    t_kpi = Table(kpi_data, colWidths=[135, 135, 135, 135])
    t_kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), c_bg_light),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 14))

    # 4. Executive Narrative Summary
    story.append(Paragraph("Executive Summary & Risk Assessment", section_h1))
    pothole_cnt = sum(cnt for cls, cnt in metrics["classes"].items() if "pothole" in cls.lower() or cls == "D40")
    crack_cnt = metrics["total"] - pothole_cnt
    summary_text = (
        f"During recent mobile dashcam and vision surveillance across the Punjab monitoring sector, "
        f"the YOLOv8 autonomous pipeline logged <b>{metrics['total']} road surface anomalies</b> "
        f"(mean AI detection confidence: <b>{metrics['avg_conf']*100:.1f}%</b>). "
        f"A total of <b>{pothole_cnt} cavities/potholes (D40)</b> and <b>{crack_cnt} structural cracks (D10/D20)</b> "
        f"were geolocated. Currently, <b>{metrics['tier1']} critical hazards</b> represent Tier 1 severe road failures "
        f"exceeding standard depth and area thresholds, posing immediate risk to vehicular safety and tire integrity."
    )
    story.append(Paragraph(summary_text, body_text))
    story.append(Spacer(1, 14))

    # 5. DBSCAN Hotspot Clusters Table
    story.append(Paragraph("DBSCAN Spatial Hotspots & High-Density Priority Zones", section_h1))
    cluster_rows = [
        [
            Paragraph("<b>Cluster ID</b>", cell_bold),
            Paragraph("<b>Center Coordinates</b>", cell_bold),
            Paragraph("<b>Hazard Count</b>", cell_bold),
            Paragraph("<b>Radius</b>", cell_bold),
            Paragraph("<b>Dominant Tier</b>", cell_bold),
            Paragraph("<b>Municipal Action</b>", cell_bold),
        ]
    ]

    if metrics["clusters"]:
        for c in metrics["clusters"][:6]:  # Show top 6 clusters
            cid = f"Hotspot #{c['cluster_id']}"
            gps = f"{c['center']['latitude']:.4f}° N, {c['center']['longitude']:.4f}° E"
            count = f"<b>{c['total_hazards']} hazards</b>"
            rad = f"~{int(c['radius_meters'])} m"
            tier = c["dominant_severity"]
            act = "Emergency Cold Patch (24h)" if tier == "Tier 1" else "Scheduled Resurfacing"

            cluster_rows.append([
                Paragraph(cid, cell_text),
                Paragraph(gps, cell_text),
                Paragraph(count, cell_text),
                Paragraph(rad, cell_text),
                Paragraph(f"<font color='{'#e11d48' if tier=='Tier 1' else '#d97706'}'><b>{tier}</b></font>", cell_text),
                Paragraph(act, cell_text),
            ])
    else:
        cluster_rows.append([
            Paragraph("No active high-density clusters (&lt;2 nearby points)", cell_text),
            Paragraph("—", cell_text),
            Paragraph(f"{metrics['total']} isolated", cell_text),
            Paragraph("—", cell_text),
            Paragraph("Tier 3", cell_text),
            Paragraph("Routine Telemetry Monitoring", cell_text),
        ])

    t_clusters = Table(cluster_rows, colWidths=[80, 130, 80, 60, 80, 110])
    t_clusters.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_bg_light]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_clusters)
    story.append(Spacer(1, 14))

    # 6. Critical Tier 1 Detections Sample
    story.append(Paragraph("Critical Hazards Awaiting Maintenance Dispatch (Tier 1 Sample)", section_h1))
    det_rows = [
        [
            Paragraph("<b>Detection ID</b>", cell_bold),
            Paragraph("<b>Class Name</b>", cell_bold),
            Paragraph("<b>Confidence</b>", cell_bold),
            Paragraph("<b>GPS Location</b>", cell_bold),
            Paragraph("<b>Timestamp (UTC)</b>", cell_bold),
        ]
    ]

    tier1_samples = [d for d in metrics["detections"] if d["severity"] == "Tier 1"][:5]
    if not tier1_samples:
        tier1_samples = metrics["detections"][:5]

    for d in tier1_samples:
        short_id = d["detection_id"][:8] if len(d["detection_id"]) > 8 else d["detection_id"]
        det_rows.append([
            Paragraph(f"<b>{short_id}</b>", cell_text),
            Paragraph(d["class_name"].upper(), cell_text),
            Paragraph(f"{d['confidence']*100:.1f}%", cell_text),
            Paragraph(f"{d['latitude']:.4f}, {d['longitude']:.4f}", cell_text),
            Paragraph(d["timestamp"][:19].replace("T", " "), cell_text),
        ])

    t_dets = Table(det_rows, colWidths=[90, 100, 70, 130, 150])
    t_dets.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_bg_light]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_dets)
    story.append(Spacer(1, 16))

    # 7. Official Sign-off & Budget Recommendation
    rec_box_data = [
        [
            Paragraph(
                "<b>Municipal Action Recommendation:</b><br/>"
                "&bull; <b>Emergency Cold-Patching:</b> Dispatch rapid-response civil teams to Hotspot #1 immediately.<br/>"
                "&bull; <b>Quarterly Budget Allocation:</b> Reserve asphalt emulsion supply for high-stress monsoon degradation.<br/>"
                "&bull; <b>Surveillance Cadence:</b> Maintain bi-weekly dashcam telemetry capture along major logistics corridors.",
                body_text,
            )
        ]
    ]
    t_rec = Table(rec_box_data, colWidths=[540])
    t_rec.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#93c5fd")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(t_rec)
    story.append(Spacer(1, 14))

    # Footer note
    story.append(Paragraph(
        "<i>The City Fixer Autonomous Road Surveillance System &bull; Powered by YOLOv8, Flask, DBSCAN, and Leaflet.js</i>",
        subtitle_style,
    ))

    # Build document
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes


if __name__ == "__main__":
    out_file = "Municipal_Road_Health_Brief.pdf"
    print(f"Generating PDF report: {out_file}...")
    pdf_data = generate_pdf_report(output_path=out_file)
    print(f"Report generated successfully! Size: {len(pdf_data):,} bytes.")
