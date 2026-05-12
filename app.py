import csv
import io
import json
import math
import os
import zipfile
from datetime import datetime
from functools import lru_cache
from html import escape

import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from utils import (
    DEFAULT_THRESHOLD,
    MAX_LENGTH,
    MAX_VARIANTS,
    MIN_LENGTH,
    classify_probability,
    encode_sequence,
    generate_substitution_library,
    generate_variants,
    input_diagnostics,
    parse_peptide_input,
    peptide_properties,
    screening_recommendation,
    sanitize_threshold,
    validate_sequence,
)


MODEL_PATH = "antifungal_peptide_model.h5"
APP_VERSION = "2026.05.12-report-exports"
DEFAULT_VARIANT_MODE = "activity_optimization"
DESIGN_ACTIONS = {"variants", "variant_download", "variant_fasta", "variant_report_pack"}
REPORT_ACTIONS = {"report_pack", "variant_report_pack"}
UTILITY_ACTIONS = {"utilities", "utility_download", "utility_report_pack"}
VARIANT_MODES = {
    "none": "Prediction only",
    "activity_optimization": "Activity-boosting substitution scan",
    "alanine_scan": "Alanine scan",
    "tryptophan_scan": "Tryptophan scan",
    "lysine_scan": "Lysine charge scan",
    "hydrophobic_tempering": "Hydrophobic tempering",
    "terminal_truncation": "Terminal truncation",
}
DESIGN_VARIANT_MODES = {
    key: label
    for key, label in VARIANT_MODES.items()
    if key != "none"
}
EXTERNAL_MODEL_CANDIDATES = [
    {
        "name": "AMPlify",
        "task": "General AMP prediction",
        "integration": "Possible, but best isolated in a legacy conda/Docker worker because upstream dependencies target Python 3.6, TensorFlow 1.12, and Keras 2.2.4.",
        "status": "candidate",
    },
    {
        "name": "AMPidentifier",
        "task": "General AMP ensemble prediction",
        "integration": "Promising Python/PyPI option with built-in RF/SVM/GB models and modlAMP descriptors; needs dependency test before adding to production requirements.",
        "status": "candidate",
    },
    {
        "name": "ToxinPred2",
        "task": "Toxicity prediction",
        "integration": "Standalone/PyPI exists, but the model bundle and optional BLAST requirements must be installed and version-locked separately.",
        "status": "candidate",
    },
    {
        "name": "HemoPI2",
        "task": "Hemolysis classification/regression",
        "integration": "GitHub/PyPI code is available, but the large model directory is distributed separately by the authors; best added later as an optional local backend.",
        "status": "not bundled",
    },
    {
        "name": "AntiFP / AntiFP2",
        "task": "Antifungal prediction",
        "integration": "AntiFP is methodologically relevant for AFPs; AntiFP2 targets antifungal proteins and may be heavier than needed for short peptide screening.",
        "status": "review",
    },
]

app = Flask(__name__)


@app.after_request
def add_no_cache_headers(response):
    if response.content_type and response.content_type.startswith("text/html"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@lru_cache(maxsize=1)
def get_model():
    from tensorflow.keras.models import load_model

    return load_model(MODEL_PATH)


def build_prediction_results(records, threshold=DEFAULT_THRESHOLD):
    results = []
    valid_entries = []
    seen_sequences = {}

    for record in records:
        diagnostics = input_diagnostics(record["sequence"])
        result = {
            "id": record["id"],
            "input_sequence": record["sequence"],
            "sequence": "",
            "rank": None,
            "probability": None,
            "probability_percent": None,
            "classification": "Invalid sequence",
            "status": "invalid",
            "alert_class": "secondary",
            "error": None,
            "diagnostics": diagnostics,
            "duplicate_of": None,
            "properties": {},
        }

        try:
            sequence = validate_sequence(record["sequence"])
            result["sequence"] = sequence
            if sequence in seen_sequences:
                result["duplicate_of"] = seen_sequences[sequence]
            else:
                seen_sequences[sequence] = record["id"]
            result["properties"] = peptide_properties(sequence)
            valid_entries.append((len(results), sequence))
        except ValueError as exc:
            result["error"] = str(exc)

        results.append(result)

    if valid_entries:
        encoded = np.vstack([encode_sequence(sequence) for _, sequence in valid_entries])
        probabilities = get_model().predict(encoded, verbose=0).reshape(-1)

        for (result_index, _), probability in zip(valid_entries, probabilities):
            probability = float(probability)
            label = classify_probability(probability, threshold)
            results[result_index].update(
                {
                    "probability": round(probability, 4),
                    "probability_percent": round(probability * 100, 2),
                    **label,
                }
            )
            results[result_index]["screening"] = screening_recommendation(
                probability,
                results[result_index]["properties"],
            )

    ranked_indexes = sorted(
        [index for index, row in enumerate(results) if row["probability"] is not None],
        key=lambda index: results[index]["probability"],
        reverse=True,
    )
    for rank, index in enumerate(ranked_indexes, start=1):
        results[index]["rank"] = rank

    return results


def summarize_results(results):
    valid_probabilities = [row["probability"] for row in results if row["probability"] is not None]
    valid_properties = [row["properties"] for row in results if row.get("properties")]
    top = max(valid_probabilities) if valid_probabilities else None
    mean = sum(valid_probabilities) / len(valid_probabilities) if valid_probabilities else None
    mean_length = sum(p["length"] for p in valid_properties) / len(valid_properties) if valid_properties else None
    mean_charge = sum(p["net_charge"] for p in valid_properties) / len(valid_properties) if valid_properties else None
    mean_hydrophobic = sum(p["hydrophobic_percent"] for p in valid_properties) / len(valid_properties) if valid_properties else None
    return {
        "total": len(results),
        "valid": sum(1 for row in results if row["status"] != "invalid"),
        "high": sum(1 for row in results if row["status"] == "high"),
        "intermediate": sum(1 for row in results if row["status"] == "intermediate"),
        "low": sum(1 for row in results if row["status"] == "low"),
        "invalid": sum(1 for row in results if row["status"] == "invalid"),
        "duplicates": sum(1 for row in results if row["duplicate_of"]),
        "advance": sum(1 for row in results if row.get("screening", {}).get("tier") == "advance"),
        "review": sum(1 for row in results if row.get("screening", {}).get("tier") == "review"),
        "deprioritize": sum(1 for row in results if row.get("screening", {}).get("tier") == "deprioritize"),
        "top_probability_percent": round(top * 100, 2) if top is not None else None,
        "mean_probability_percent": round(mean * 100, 2) if mean is not None else None,
        "mean_length": round(mean_length, 1) if mean_length is not None else None,
        "mean_charge": round(mean_charge, 2) if mean_charge is not None else None,
        "mean_hydrophobic_percent": round(mean_hydrophobic, 1) if mean_hydrophobic is not None else None,
    }


def top_valid_result(results):
    valid = [row for row in results if row["probability"] is not None]
    if not valid:
        return None
    return sorted(valid, key=lambda row: row["probability"], reverse=True)[0]


def summarize_variant_positions(variant_results):
    best_by_position = {}
    for row in variant_results:
        position = row.get("position")
        if position is None or row.get("probability") is None:
            continue
        current = best_by_position.get(position)
        if current is None or row["probability"] > current["probability"]:
            best_by_position[position] = row

    summary = []
    for position in sorted(best_by_position):
        row = best_by_position[position]
        summary.append(
            {
                "position": position,
                "change": row.get("change"),
                "sequence": row.get("sequence"),
                "probability_percent": row.get("probability_percent"),
                "delta_probability_percent": row.get("delta_probability_percent"),
                "classification": row.get("classification"),
            }
        )
    return summary


def build_variant_results(parent_result, mode, threshold=DEFAULT_THRESHOLD):
    if not parent_result or mode == "none":
        return [], []
    if mode == "activity_optimization":
        variants = generate_substitution_library(parent_result["sequence"])
    else:
        variants = generate_variants(parent_result["sequence"], mode=mode, max_variants=MAX_VARIANTS)
    if not variants:
        return [], []
    records = [{"id": row["id"], "sequence": row["sequence"]} for row in variants]
    variant_results = build_prediction_results(records, threshold)
    metadata_by_id = {row["id"]: row for row in variants}

    for row in variant_results:
        metadata = metadata_by_id.get(row["id"], {})
        row["change"] = metadata.get("change", "")
        row["rationale"] = metadata.get("rationale", "")
        row["position"] = metadata.get("position")
        row["from"] = metadata.get("from")
        row["to"] = metadata.get("to")
        row["parent_id"] = parent_result["id"]
        row["parent_sequence"] = parent_result["sequence"]
        row["delta_probability_percent"] = None
        if row["probability"] is not None and parent_result["probability"] is not None:
            row["delta_probability_percent"] = round(
                (row["probability"] - parent_result["probability"]) * 100,
                2,
            )

    variant_results = sorted(
        variant_results,
        key=lambda row: row["probability"] if row["probability"] is not None else -1,
        reverse=True,
    )
    position_summary = summarize_variant_positions(variant_results)

    if mode == "activity_optimization":
        improving = [
            row
            for row in variant_results
            if row.get("delta_probability_percent") is not None and row["delta_probability_percent"] > 0
        ]
        variant_results = improving[:MAX_VARIANTS] if improving else variant_results[:MAX_VARIANTS]

    return variant_results, position_summary


def results_to_csv(results):
    output = io.StringIO()
    fieldnames = [
        "id",
        "sequence",
        "probability",
        "probability_percent",
        "classification",
        "status",
        "rank",
        "duplicate_of",
        "screening_tier",
        "afp_feature_score",
        "liabilities",
        "length",
        "formula",
        "molecular_weight",
        "net_charge",
        "charge_ph_5_5",
        "charge_ph_7_0",
        "charge_ph_7_4",
        "charge_density",
        "isoelectric_point",
        "gravy",
        "aliphatic_index",
        "hydrophobic_moment",
        "hydrophobic_percent",
        "positive_percent",
        "negative_percent",
        "polar_percent",
        "aromaticity",
        "sequence_entropy",
        "extinction_reduced",
        "extinction_oxidized",
        "absorbance_0_1_percent_reduced",
        "absorbance_0_1_percent_oxidized",
        "cysteines",
        "notes",
        "alerts",
        "error",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for row in results:
        properties = row.get("properties") or {}
        screening = row.get("screening") or {}
        writer.writerow(
            {
                "id": row["id"],
                "sequence": row["sequence"] or row["input_sequence"],
                "probability": row["probability"],
                "probability_percent": row["probability_percent"],
                "classification": row["classification"],
                "status": row["status"],
                "rank": row["rank"],
                "duplicate_of": row["duplicate_of"],
                "screening_tier": screening.get("tier"),
                "afp_feature_score": screening.get("afp_feature_score"),
                "liabilities": screening.get("liabilities"),
                "length": properties.get("length"),
                "formula": properties.get("formula"),
                "molecular_weight": properties.get("molecular_weight"),
                "net_charge": properties.get("net_charge"),
                "charge_ph_5_5": properties.get("charge_ph_5_5"),
                "charge_ph_7_0": properties.get("charge_ph_7_0"),
                "charge_ph_7_4": properties.get("charge_ph_7_4"),
                "charge_density": properties.get("charge_density"),
                "isoelectric_point": properties.get("isoelectric_point"),
                "gravy": properties.get("gravy"),
                "aliphatic_index": properties.get("aliphatic_index"),
                "hydrophobic_moment": properties.get("hydrophobic_moment"),
                "hydrophobic_percent": properties.get("hydrophobic_percent"),
                "positive_percent": properties.get("positive_percent"),
                "negative_percent": properties.get("negative_percent"),
                "polar_percent": properties.get("polar_percent"),
                "aromaticity": properties.get("aromaticity"),
                "sequence_entropy": properties.get("sequence_entropy"),
                "extinction_reduced": properties.get("extinction_reduced"),
                "extinction_oxidized": properties.get("extinction_oxidized"),
                "absorbance_0_1_percent_reduced": properties.get("absorbance_0_1_percent_reduced"),
                "absorbance_0_1_percent_oxidized": properties.get("absorbance_0_1_percent_oxidized"),
                "cysteines": properties.get("cysteines"),
                "notes": properties.get("notes"),
                "alerts": " | ".join(alert["title"] for alert in properties.get("alerts", [])),
                "error": row["error"],
            }
        )

    return output.getvalue()


def fasta_safe(value):
    return str(value).replace("|", "/").replace("\n", " ").strip()


def wrap_fasta_sequence(sequence, width=60):
    return "\n".join(sequence[index : index + width] for index in range(0, len(sequence), width))


def results_to_fasta(results, include_variants=False):
    lines = []
    for row in results:
        sequence = row.get("sequence")
        if not sequence:
            continue

        header_parts = [fasta_safe(row["id"])]
        if row.get("rank"):
            header_parts.append(f"rank={row['rank']}")
        if row.get("probability_percent") is not None:
            header_parts.append(f"score={row['probability_percent']}pct")
        if row.get("classification"):
            header_parts.append(f"class={fasta_safe(row['classification'])}")
        if row.get("screening"):
            header_parts.append(f"tier={fasta_safe(row['screening'].get('tier'))}")
        if include_variants:
            if row.get("change"):
                header_parts.append(f"change={fasta_safe(row['change'])}")
            if row.get("delta_probability_percent") is not None:
                header_parts.append(f"delta_pp={row['delta_probability_percent']}")
            if row.get("parent_id"):
                header_parts.append(f"parent={fasta_safe(row['parent_id'])}")

        lines.append(">" + "|".join(header_parts))
        lines.append(wrap_fasta_sequence(sequence))

    return "\n".join(lines) + ("\n" if lines else "")


def variant_results_to_csv(parent_result, variant_results):
    output = io.StringIO()
    fieldnames = [
        "parent_id",
        "parent_sequence",
        "parent_probability_percent",
        "id",
        "sequence",
        "change",
        "position",
        "from",
        "to",
        "rationale",
        "probability",
        "probability_percent",
        "delta_probability_percent",
        "classification",
        "status",
        "screening_tier",
        "afp_feature_score",
        "liabilities",
        "length",
        "molecular_weight",
        "net_charge",
        "isoelectric_point",
        "gravy",
        "hydrophobic_moment",
        "hydrophobic_percent",
        "positive_percent",
        "chemical_liabilities",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for row in variant_results:
        properties = row.get("properties") or {}
        screening = row.get("screening") or {}
        writer.writerow(
            {
                "parent_id": parent_result.get("id") if parent_result else row.get("parent_id"),
                "parent_sequence": parent_result.get("sequence") if parent_result else row.get("parent_sequence"),
                "parent_probability_percent": parent_result.get("probability_percent") if parent_result else None,
                "id": row.get("id"),
                "sequence": row.get("sequence"),
                "change": row.get("change"),
                "position": row.get("position"),
                "from": row.get("from"),
                "to": row.get("to"),
                "rationale": row.get("rationale"),
                "probability": row.get("probability"),
                "probability_percent": row.get("probability_percent"),
                "delta_probability_percent": row.get("delta_probability_percent"),
                "classification": row.get("classification"),
                "status": row.get("status"),
                "screening_tier": screening.get("tier"),
                "afp_feature_score": screening.get("afp_feature_score"),
                "liabilities": screening.get("liabilities"),
                "length": properties.get("length"),
                "molecular_weight": properties.get("molecular_weight"),
                "net_charge": properties.get("net_charge"),
                "isoelectric_point": properties.get("isoelectric_point"),
                "gravy": properties.get("gravy"),
                "hydrophobic_moment": properties.get("hydrophobic_moment"),
                "hydrophobic_percent": properties.get("hydrophobic_percent"),
                "positive_percent": properties.get("positive_percent"),
                "chemical_liabilities": " | ".join(
                    item["title"] for item in properties.get("chemical_liabilities", [])
                ),
            }
        )

    return output.getvalue()


PLOT_STATUS_COLORS = {
    "high": "#06d6a0",
    "intermediate": "#ffd166",
    "low": "#ef476f",
    "invalid": "#94a8bd",
    "valid": "#118ab2",
}


def clean_plot_label(value, max_length=34):
    label = str(value or "").strip()
    if len(label) > max_length:
        label = f"{label[: max_length - 1]}..."
    return escape(label, quote=True)


def valid_rows(rows):
    return [row for row in rows if row.get("properties")]


def no_data_svg(title, detail="No valid rows available for this plot.", width=880, height=260):
    title = escape(title, quote=True)
    detail = escape(detail, quote=True)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
  <rect width="100%" height="100%" rx="12" fill="#10192a"/>
  <text x="40" y="70" fill="#edf4fb" font-family="Arial, sans-serif" font-size="26" font-weight="700">{title}</text>
  <text x="40" y="115" fill="#94a8bd" font-family="Arial, sans-serif" font-size="16">{detail}</text>
</svg>
"""


def plot_shell(title, subtitle, width, height, body):
    title = escape(title, quote=True)
    subtitle = escape(subtitle, quote=True)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
  <rect width="100%" height="100%" rx="12" fill="#10192a"/>
  <text x="34" y="42" fill="#edf4fb" font-family="Arial, sans-serif" font-size="24" font-weight="700">{title}</text>
  <text x="34" y="68" fill="#94a8bd" font-family="Arial, sans-serif" font-size="13">{subtitle}</text>
  {body}
</svg>
"""


def score_ranking_svg(results, threshold):
    rows = sorted(
        [row for row in results if row.get("probability_percent") is not None],
        key=lambda row: row["probability_percent"],
        reverse=True,
    )[:25]
    if not rows:
        return no_data_svg("AFPRED score ranking")

    width = 980
    top = 92
    row_gap = 30
    height = top + len(rows) * row_gap + 44
    left = 205
    right = 55
    plot_width = width - left - right
    threshold_x = left + plot_width * threshold
    parts = [
        f'<line x1="{left}" y1="{top - 18}" x2="{left + plot_width}" y2="{top - 18}" stroke="#26364f"/>',
        f'<line x1="{threshold_x:.1f}" y1="{top - 28}" x2="{threshold_x:.1f}" y2="{height - 38}" stroke="#edf4fb" stroke-dasharray="5 5" opacity="0.65"/>',
        f'<text x="{threshold_x + 5:.1f}" y="{top - 28}" fill="#edf4fb" font-family="Arial, sans-serif" font-size="11">threshold {threshold:.2f}</text>',
    ]
    for tick in [0, 25, 50, 75, 100]:
        x = left + plot_width * tick / 100
        parts.append(f'<line x1="{x:.1f}" y1="{top - 23}" x2="{x:.1f}" y2="{height - 38}" stroke="#26364f" opacity="0.55"/>')
        parts.append(f'<text x="{x - 8:.1f}" y="{height - 18}" fill="#94a8bd" font-family="Arial, sans-serif" font-size="11">{tick}</text>')

    for index, row in enumerate(rows):
        y = top + index * row_gap
        score = row["probability_percent"]
        bar_width = max(2, plot_width * score / 100)
        color = PLOT_STATUS_COLORS.get(row.get("status"), "#118ab2")
        parts.append(f'<text x="34" y="{y + 14}" fill="#edf4fb" font-family="Arial, sans-serif" font-size="12">{clean_plot_label(row.get("id"))}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="18" rx="5" fill="{color}"/>')
        parts.append(f'<text x="{left + bar_width + 8:.1f}" y="{y + 14}" fill="#edf4fb" font-family="Arial, sans-serif" font-size="12">{score:.2f}%</text>')

    return plot_shell(
        "AFPRED score ranking",
        "Top valid peptides ranked by predicted antifungal probability.",
        width,
        height,
        "\n  ".join(parts),
    )


def score_distribution_svg(results):
    scores = [row["probability_percent"] for row in results if row.get("probability_percent") is not None]
    if not scores:
        return no_data_svg("Score distribution")

    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100.0001)]
    counts = []
    for low, high in bins:
        counts.append(sum(1 for score in scores if low <= score < high))
    max_count = max(counts) or 1
    width = 880
    height = 420
    left = 82
    top = 86
    plot_width = width - left - 42
    plot_height = height - top - 70
    bar_gap = 18
    bar_width = (plot_width - bar_gap * (len(bins) - 1)) / len(bins)
    parts = [
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#26364f"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#26364f"/>',
    ]
    for tick in range(0, max_count + 1):
        if max_count > 8 and tick % max(1, math.ceil(max_count / 5)) != 0:
            continue
        y = top + plot_height - (plot_height * tick / max_count)
        parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#26364f" opacity="0.45"/>')
        parts.append(f'<text x="42" y="{y + 4:.1f}" fill="#94a8bd" font-family="Arial, sans-serif" font-size="11">{tick}</text>')

    for index, ((low, high), count) in enumerate(zip(bins, counts)):
        x = left + index * (bar_width + bar_gap)
        bar_height = plot_height * count / max_count
        y = top + plot_height - bar_height
        label_high = int(high if high <= 100 else 100)
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="7" fill="#118ab2"/>')
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" fill="#edf4fb" font-family="Arial, sans-serif" font-size="13" text-anchor="middle">{count}</text>')
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{height - 28}" fill="#94a8bd" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">{int(low)}-{label_high}%</text>')

    return plot_shell(
        "Score distribution",
        "Candidate count by AFPRED probability band.",
        width,
        height,
        "\n  ".join(parts),
    )


def property_map_svg(results):
    rows = valid_rows(results)
    if not rows:
        return no_data_svg("Charge-hydrophobicity map")

    charges = [row["properties"]["net_charge"] for row in rows]
    min_charge = math.floor(min(charges + [0]) - 1)
    max_charge = math.ceil(max(charges + [0]) + 1)
    if min_charge == max_charge:
        min_charge -= 1
        max_charge += 1

    width = 900
    height = 520
    left = 90
    top = 86
    plot_width = width - left - 58
    plot_height = height - top - 78

    def x_scale(value):
        return left + plot_width * max(0, min(100, value)) / 100

    def y_scale(value):
        return top + plot_height - plot_height * (value - min_charge) / (max_charge - min_charge)

    parts = [
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#26364f"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#26364f"/>',
        f'<text x="{left + plot_width / 2:.1f}" y="{height - 26}" fill="#94a8bd" font-family="Arial, sans-serif" font-size="13" text-anchor="middle">Hydrophobic residues (%)</text>',
        f'<text x="24" y="{top + plot_height / 2:.1f}" fill="#94a8bd" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 24 {top + plot_height / 2:.1f})" text-anchor="middle">Net charge</text>',
    ]
    for tick in [0, 25, 50, 75, 100]:
        x = x_scale(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#26364f" opacity="0.35"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_height + 22}" fill="#94a8bd" font-family="Arial, sans-serif" font-size="11" text-anchor="middle">{tick}</text>')
    for tick in range(min_charge, max_charge + 1):
        if max_charge - min_charge > 12 and tick % 2:
            continue
        y = y_scale(tick)
        parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#26364f" opacity="0.35"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.1f}" fill="#94a8bd" font-family="Arial, sans-serif" font-size="11" text-anchor="end">{tick}</text>')

    labeled = sorted(rows, key=lambda row: row.get("probability_percent") or 0, reverse=True)[:10]
    labeled_ids = {row.get("id") for row in labeled}
    for row in rows:
        properties = row["properties"]
        x = x_scale(properties["hydrophobic_percent"])
        y = y_scale(properties["net_charge"])
        color = PLOT_STATUS_COLORS.get(row.get("status"), "#118ab2")
        radius = 5 + min(7, max(0, (row.get("probability_percent") or 50) / 16))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.78" stroke="#edf4fb" stroke-opacity="0.28"/>')
        if row.get("id") in labeled_ids:
            parts.append(f'<text x="{x + 9:.1f}" y="{y - 8:.1f}" fill="#edf4fb" font-family="Arial, sans-serif" font-size="11">{clean_plot_label(row.get("id"), 22)}</text>')

    return plot_shell(
        "Charge-hydrophobicity map",
        "Descriptor space for valid peptides; point color follows prediction class when available.",
        width,
        height,
        "\n  ".join(parts),
    )


def charge_profiles_svg(results):
    rows = valid_rows(results)[:8]
    if not rows:
        return no_data_svg("pH charge profiles")

    points = rows[0]["properties"].get("charge_profile", [])
    if not points:
        return no_data_svg("pH charge profiles", "No charge profile values are available.")

    all_charges = [
        point["charge"]
        for row in rows
        for point in row["properties"].get("charge_profile", [])
    ]
    min_charge = math.floor(min(all_charges + [0]) - 1)
    max_charge = math.ceil(max(all_charges + [0]) + 1)
    min_ph = min(point["ph"] for point in points)
    max_ph = max(point["ph"] for point in points)
    width = 900
    height = 500
    left = 88
    top = 86
    plot_width = width - left - 170
    plot_height = height - top - 78
    palette = ["#06d6a0", "#118ab2", "#ffd166", "#ef476f", "#9b8cff", "#5dd9c1", "#f78c6b", "#b8e986"]

    def x_scale(value):
        return left + plot_width * (value - min_ph) / (max_ph - min_ph)

    def y_scale(value):
        return top + plot_height - plot_height * (value - min_charge) / (max_charge - min_charge)

    parts = [
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#26364f"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#26364f"/>',
        f'<text x="{left + plot_width / 2:.1f}" y="{height - 26}" fill="#94a8bd" font-family="Arial, sans-serif" font-size="13" text-anchor="middle">pH</text>',
        f'<text x="24" y="{top + plot_height / 2:.1f}" fill="#94a8bd" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 24 {top + plot_height / 2:.1f})" text-anchor="middle">Estimated charge</text>',
    ]
    for point in points:
        x = x_scale(point["ph"])
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#26364f" opacity="0.35"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_height + 22}" fill="#94a8bd" font-family="Arial, sans-serif" font-size="11" text-anchor="middle">{point["ph"]}</text>')
    for tick in range(min_charge, max_charge + 1):
        y = y_scale(tick)
        parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#26364f" opacity="0.3"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.1f}" fill="#94a8bd" font-family="Arial, sans-serif" font-size="11" text-anchor="end">{tick}</text>')

    for index, row in enumerate(rows):
        color = palette[index % len(palette)]
        profile = row["properties"].get("charge_profile", [])
        coords = " ".join(f'{x_scale(point["ph"]):.1f},{y_scale(point["charge"]):.1f}' for point in profile)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        for point in profile:
            parts.append(f'<circle cx="{x_scale(point["ph"]):.1f}" cy="{y_scale(point["charge"]):.1f}" r="3.5" fill="{color}"/>')
        legend_y = top + index * 24
        parts.append(f'<rect x="{left + plot_width + 30}" y="{legend_y - 10}" width="12" height="12" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{left + plot_width + 50}" y="{legend_y}" fill="#edf4fb" font-family="Arial, sans-serif" font-size="12">{clean_plot_label(row.get("id"), 20)}</text>')

    return plot_shell(
        "pH charge profiles",
        "Estimated peptide charge across assay-relevant pH values.",
        width,
        height,
        "\n  ".join(parts),
    )


def variant_delta_svg(variant_results):
    rows = [
        row
        for row in variant_results
        if row.get("delta_probability_percent") is not None
    ]
    rows = sorted(rows, key=lambda row: row["delta_probability_percent"], reverse=True)[:25]
    if not rows:
        return no_data_svg("Variant score delta", "Run a variant analysis to generate delta values.")

    min_delta = min([0] + [row["delta_probability_percent"] for row in rows])
    max_delta = max([0] + [row["delta_probability_percent"] for row in rows])
    if min_delta == max_delta:
        min_delta -= 1
        max_delta += 1

    width = 980
    top = 88
    row_gap = 30
    height = top + len(rows) * row_gap + 48
    left = 215
    right = 65
    plot_width = width - left - right

    def x_scale(value):
        return left + plot_width * (value - min_delta) / (max_delta - min_delta)

    zero_x = x_scale(0)
    parts = [
        f'<line x1="{zero_x:.1f}" y1="{top - 22}" x2="{zero_x:.1f}" y2="{height - 38}" stroke="#edf4fb" stroke-dasharray="5 5" opacity="0.65"/>',
        f'<text x="{zero_x + 5:.1f}" y="{top - 27}" fill="#edf4fb" font-family="Arial, sans-serif" font-size="11">parent score</text>',
    ]
    for row_index, row in enumerate(rows):
        y = top + row_index * row_gap
        delta = row["delta_probability_percent"]
        x = x_scale(delta)
        bar_x = min(zero_x, x)
        bar_width = max(2, abs(x - zero_x))
        color = "#06d6a0" if delta >= 0 else "#ef476f"
        parts.append(f'<text x="34" y="{y + 14}" fill="#edf4fb" font-family="Arial, sans-serif" font-size="12">{clean_plot_label(row.get("change") or row.get("id"))}</text>')
        parts.append(f'<rect x="{bar_x:.1f}" y="{y}" width="{bar_width:.1f}" height="18" rx="5" fill="{color}"/>')
        parts.append(f'<text x="{x + (8 if delta >= 0 else -44):.1f}" y="{y + 14}" fill="#edf4fb" font-family="Arial, sans-serif" font-size="12">{delta:+.2f} pp</text>')

    return plot_shell(
        "Variant score delta",
        "Single-variant score change versus the selected parent peptide.",
        width,
        height,
        "\n  ".join(parts),
    )


def build_prediction_plots(results, threshold, variant_results=None):
    plots = {
        "score_ranking.svg": score_ranking_svg(results, threshold),
        "score_distribution.svg": score_distribution_svg(results),
        "charge_hydrophobicity_map.svg": property_map_svg(results),
        "charge_profiles.svg": charge_profiles_svg(results),
    }
    if variant_results:
        plots["variant_delta.svg"] = variant_delta_svg(variant_results)
    return plots


def report_metric(label, value):
    if value is None:
        value = "--"
    return f'<div class="metric"><span>{escape(str(label))}</span><strong>{escape(str(value))}</strong></div>'


def report_table(rows, columns):
    if not rows:
        return "<p>No rows available.</p>"
    header = "".join(f"<th>{escape(label)}</th>" for label, _ in columns)
    body_rows = []
    for row in rows:
        cells = []
        for _, getter in columns:
            value = getter(row)
            cells.append(f"<td>{escape(str(value if value is not None else '--'))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def build_prediction_report_html(
    results,
    summary,
    threshold,
    plots,
    variant_parent=None,
    variant_results=None,
    variant_position_summary=None,
    variant_mode=None,
):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    valid = sorted(
        [row for row in results if row.get("probability_percent") is not None],
        key=lambda row: row["probability_percent"],
        reverse=True,
    )
    top_rows = valid[:20]
    summary = summary or summarize_results(results)
    plot_sections = "".join(
        f"<section><h2>{escape(filename.replace('_', ' ').replace('.svg', '').title())}</h2>{svg}</section>"
        for filename, svg in plots.items()
    )
    metrics = "".join(
        [
            report_metric("Total records", summary.get("total")),
            report_metric("Valid", summary.get("valid")),
            report_metric("Likely AFP", summary.get("high")),
            report_metric("Review", summary.get("intermediate")),
            report_metric("Low signal", summary.get("low")),
            report_metric("Invalid", summary.get("invalid")),
            report_metric("Advance tier", summary.get("advance")),
            report_metric("Top score %", summary.get("top_probability_percent")),
            report_metric("Mean charge", summary.get("mean_charge")),
            report_metric("Mean hydrophobic %", summary.get("mean_hydrophobic_percent")),
        ]
    )
    top_table = report_table(
        top_rows,
        [
            ("Rank", lambda row: row.get("rank")),
            ("ID", lambda row: row.get("id")),
            ("Sequence", lambda row: row.get("sequence")),
            ("Score %", lambda row: row.get("probability_percent")),
            ("Class", lambda row: row.get("classification")),
            ("Tier", lambda row: (row.get("screening") or {}).get("tier")),
            ("Charge", lambda row: row.get("properties", {}).get("net_charge")),
            ("Hydrophobic %", lambda row: row.get("properties", {}).get("hydrophobic_percent")),
        ],
    )

    variant_section = ""
    if variant_results:
        variant_label = VARIANT_MODES.get(variant_mode, variant_mode or "variant analysis")
        variant_table = report_table(
            variant_results[:20],
            [
                ("Change", lambda row: row.get("change")),
                ("Sequence", lambda row: row.get("sequence")),
                ("Score %", lambda row: row.get("probability_percent")),
                ("Delta pp", lambda row: row.get("delta_probability_percent")),
                ("Class", lambda row: row.get("classification")),
                ("Rationale", lambda row: row.get("rationale")),
            ],
        )
        position_table = report_table(
            variant_position_summary or [],
            [
                ("Position", lambda row: row.get("position")),
                ("Best change", lambda row: row.get("change")),
                ("Score %", lambda row: row.get("probability_percent")),
                ("Delta pp", lambda row: row.get("delta_probability_percent")),
                ("Class", lambda row: row.get("classification")),
            ],
        )
        variant_section = f"""
        <section>
          <h2>Variant Analysis</h2>
          <p><strong>Mode:</strong> {escape(str(variant_label))}</p>
          <p><strong>Parent:</strong> {escape(str(variant_parent.get("id") if variant_parent else "--"))}
          | {escape(str(variant_parent.get("sequence") if variant_parent else "--"))}</p>
          <h3>Top variants</h3>
          {variant_table}
          <h3>Best mutation by position</h3>
          {position_table}
        </section>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AFPRED Screening Report</title>
  <style>
    body {{ margin: 0; padding: 28px; background: #08111f; color: #edf4fb; font-family: Arial, sans-serif; }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    section {{ margin: 18px 0; padding: 18px; border: 1px solid #26364f; border-radius: 10px; background: #10192a; }}
    h1, h2, h3 {{ margin-top: 0; }}
    p {{ color: #b8c7d6; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
    .metric {{ padding: 12px; border: 1px solid #26364f; border-radius: 8px; background: #151f33; }}
    .metric span {{ display: block; color: #94a8bd; font-size: 12px; text-transform: uppercase; }}
    .metric strong {{ display: block; margin-top: 5px; font-size: 22px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px; border-bottom: 1px solid #26364f; text-align: left; vertical-align: top; }}
    th {{ color: #06d6a0; }}
    svg {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>AFPRED Screening Report</h1>
      <p>Generated {escape(generated_at)}. Threshold: {threshold:.2f}. This report is intended for screening, prioritization, and experimental planning.</p>
    </section>
    <section>
      <h2>Batch Summary</h2>
      <div class="metrics">{metrics}</div>
    </section>
    {plot_sections}
    <section>
      <h2>Top Candidate Table</h2>
      {top_table}
    </section>
    {variant_section}
    <section>
      <h2>Scope Note</h2>
      <p>AFPRED is a sequence-based screening aid. Candidate advancement should include experimental antifungal validation, toxicity or hemolysis assessment, solubility and stability testing, novelty checks, and expert review.</p>
    </section>
  </main>
</body>
</html>
"""


def build_report_pack(results, summary, threshold, variant_parent=None, variant_results=None, position_summary=None, variant_mode=None):
    plots = build_prediction_plots(results, threshold, variant_results)
    report_html = build_prediction_report_html(
        results,
        summary,
        threshold,
        plots,
        variant_parent=variant_parent,
        variant_results=variant_results,
        variant_position_summary=position_summary,
        variant_mode=variant_mode,
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("afpred_report.html", report_html)
        archive.writestr("data/afpred_predictions.csv", results_to_csv(results))
        archive.writestr("data/afpred_predictions.fasta", results_to_fasta(results))
        for filename, svg in plots.items():
            archive.writestr(f"plots/{filename}", svg)
        if variant_results:
            archive.writestr("data/afpred_variants.csv", variant_results_to_csv(variant_parent, variant_results))
            archive.writestr("data/afpred_variants.fasta", results_to_fasta(variant_results, include_variants=True))
    output.seek(0)
    return output.getvalue()


def build_utility_report_html(results, summary, plots):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    valid = [row for row in results if row.get("status") == "valid"]
    metrics = "".join(
        [
            report_metric("Total records", summary.get("total")),
            report_metric("Valid", summary.get("valid")),
            report_metric("Invalid", summary.get("invalid")),
            report_metric("Mean mass", summary.get("mean_mass")),
            report_metric("Mean pI", summary.get("mean_pi")),
            report_metric("Mean charge", summary.get("mean_charge")),
        ]
    )
    plot_sections = "".join(
        f"<section><h2>{escape(filename.replace('_', ' ').replace('.svg', '').title())}</h2>{svg}</section>"
        for filename, svg in plots.items()
    )
    utility_table = report_table(
        valid[:30],
        [
            ("ID", lambda row: row.get("id")),
            ("Sequence", lambda row: row.get("sequence")),
            ("Mass", lambda row: row.get("properties", {}).get("molecular_weight")),
            ("pI", lambda row: row.get("properties", {}).get("isoelectric_point")),
            ("Charge", lambda row: row.get("properties", {}).get("net_charge")),
            ("GRAVY", lambda row: row.get("properties", {}).get("gravy")),
            ("Hydrophobic %", lambda row: row.get("properties", {}).get("hydrophobic_percent")),
            ("Notes", lambda row: row.get("properties", {}).get("notes")),
        ],
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AFPRED Peptide Utility Report</title>
  <style>
    body {{ margin: 0; padding: 28px; background: #08111f; color: #edf4fb; font-family: Arial, sans-serif; }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    section {{ margin: 18px 0; padding: 18px; border: 1px solid #26364f; border-radius: 10px; background: #10192a; }}
    h1, h2 {{ margin-top: 0; }}
    p {{ color: #b8c7d6; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
    .metric {{ padding: 12px; border: 1px solid #26364f; border-radius: 8px; background: #151f33; }}
    .metric span {{ display: block; color: #94a8bd; font-size: 12px; text-transform: uppercase; }}
    .metric strong {{ display: block; margin-top: 5px; font-size: 22px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px; border-bottom: 1px solid #26364f; text-align: left; vertical-align: top; }}
    th {{ color: #06d6a0; }}
    svg {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>AFPRED Peptide Utility Report</h1>
      <p>Generated {escape(generated_at)}. These calculations assume linear canonical peptide sequences.</p>
    </section>
    <section>
      <h2>Utility Summary</h2>
      <div class="metrics">{metrics}</div>
    </section>
    {plot_sections}
    <section>
      <h2>Peptide Descriptor Table</h2>
      {utility_table}
    </section>
  </main>
</body>
</html>
"""


def build_utility_report_pack(results, summary):
    plots = {
        "charge_hydrophobicity_map.svg": property_map_svg(results),
        "charge_profiles.svg": charge_profiles_svg(results),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("peptide_utility_report.html", build_utility_report_html(results, summary, plots))
        archive.writestr("data/peptide_utilities.csv", utility_results_to_csv(results))
        for filename, svg in plots.items():
            archive.writestr(f"plots/{filename}", svg)
    output.seek(0)
    return output.getvalue()


def build_utility_results(records):
    results = []
    for record in records:
        diagnostics = input_diagnostics(record["sequence"])
        result = {
            "id": record["id"],
            "input_sequence": record["sequence"],
            "sequence": "",
            "status": "invalid",
            "error": None,
            "diagnostics": diagnostics,
            "properties": {},
        }
        try:
            sequence = validate_sequence(record["sequence"])
            result["sequence"] = sequence
            result["status"] = "valid"
            result["properties"] = peptide_properties(sequence)
        except ValueError as exc:
            result["error"] = str(exc)
        results.append(result)
    return results


def summarize_utility_results(results):
    valid = [row for row in results if row["status"] == "valid"]
    if not valid:
        return {
            "total": len(results),
            "valid": 0,
            "invalid": len(results),
            "mean_mass": None,
            "mean_pi": None,
            "mean_charge": None,
        }
    return {
        "total": len(results),
        "valid": len(valid),
        "invalid": len(results) - len(valid),
        "mean_mass": round(sum(row["properties"]["molecular_weight"] for row in valid) / len(valid), 2),
        "mean_pi": round(sum(row["properties"]["isoelectric_point"] for row in valid) / len(valid), 2),
        "mean_charge": round(sum(row["properties"]["net_charge"] for row in valid) / len(valid), 2),
    }


def utility_results_to_csv(results):
    output = io.StringIO()
    fieldnames = [
        "id",
        "sequence",
        "status",
        "length",
        "formula",
        "molecular_weight",
        "isoelectric_point",
        "net_charge",
        "charge_ph_5_5",
        "charge_ph_7_0",
        "charge_ph_7_4",
        "gravy",
        "hydrophobic_moment",
        "hydrophobic_percent",
        "aliphatic_index",
        "normalized_hydrophobic_moment_kd",
        "eisenberg_hydrophobicity",
        "eisenberg_hydrophobic_moment",
        "heliquest_like_discriminant",
        "heliquest_like_lipid_binding",
        "heliquest_like_transmembrane",
        "hydrophobic_face_angle",
        "linear_hydrophobic_moment",
        "linear_moment_eisenberg",
        "boman_index",
        "disorder_promoting_percent",
        "max_hydrophobic_run",
        "max_beta_aggregation_run",
        "aggregation_risk_score",
        "basic_residue_count",
        "acidic_residue_count",
        "basic_to_acidic_ratio",
        "extinction_reduced",
        "extinction_oxidized",
        "absorbance_0_1_percent_reduced",
        "absorbance_0_1_percent_oxidized",
        "hydrophobic_hotspot",
        "charge_profile",
        "notes",
        "chemical_liabilities",
        "error",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        properties = row.get("properties") or {}
        writer.writerow(
            {
                "id": row["id"],
                "sequence": row["sequence"] or row["input_sequence"],
                "status": row["status"],
                "length": properties.get("length"),
                "formula": properties.get("formula"),
                "molecular_weight": properties.get("molecular_weight"),
                "isoelectric_point": properties.get("isoelectric_point"),
                "net_charge": properties.get("net_charge"),
                "charge_ph_5_5": properties.get("charge_ph_5_5"),
                "charge_ph_7_0": properties.get("charge_ph_7_0"),
                "charge_ph_7_4": properties.get("charge_ph_7_4"),
                "gravy": properties.get("gravy"),
                "hydrophobic_moment": properties.get("hydrophobic_moment"),
                "hydrophobic_percent": properties.get("hydrophobic_percent"),
                "aliphatic_index": properties.get("aliphatic_index"),
                "normalized_hydrophobic_moment_kd": properties.get("normalized_hydrophobic_moment_kd"),
                "eisenberg_hydrophobicity": properties.get("eisenberg_hydrophobicity"),
                "eisenberg_hydrophobic_moment": properties.get("eisenberg_hydrophobic_moment"),
                "heliquest_like_discriminant": properties.get("heliquest_like_discriminant"),
                "heliquest_like_lipid_binding": properties.get("heliquest_like_lipid_binding"),
                "heliquest_like_transmembrane": properties.get("heliquest_like_transmembrane"),
                "hydrophobic_face_angle": properties.get("hydrophobic_face_angle"),
                "linear_hydrophobic_moment": properties.get("linear_hydrophobic_moment"),
                "linear_moment_eisenberg": properties.get("linear_moment_eisenberg"),
                "boman_index": properties.get("boman_index"),
                "disorder_promoting_percent": properties.get("disorder_promoting_percent"),
                "max_hydrophobic_run": properties.get("max_hydrophobic_run"),
                "max_beta_aggregation_run": properties.get("max_beta_aggregation_run"),
                "aggregation_risk_score": properties.get("aggregation_risk_score"),
                "basic_residue_count": properties.get("basic_residue_count"),
                "acidic_residue_count": properties.get("acidic_residue_count"),
                "basic_to_acidic_ratio": properties.get("basic_to_acidic_ratio"),
                "extinction_reduced": properties.get("extinction_reduced"),
                "extinction_oxidized": properties.get("extinction_oxidized"),
                "absorbance_0_1_percent_reduced": properties.get("absorbance_0_1_percent_reduced"),
                "absorbance_0_1_percent_oxidized": properties.get("absorbance_0_1_percent_oxidized"),
                "hydrophobic_hotspot": json.dumps(properties.get("hydrophobic_hotspot") or {}),
                "charge_profile": json.dumps(properties.get("charge_profile") or []),
                "notes": properties.get("notes"),
                "chemical_liabilities": " | ".join(item["title"] for item in properties.get("chemical_liabilities", [])),
                "error": row["error"],
            }
        )
    return output.getvalue()


@app.route("/", methods=["GET", "POST"])
def index():
    raw_sequences = ""
    utility_raw_sequences = ""
    threshold = DEFAULT_THRESHOLD
    variant_mode = DEFAULT_VARIANT_MODE
    results = []
    variant_results = []
    variant_position_summary = []
    variant_parent = None
    utility_results = []
    utility_summary = None
    summary = None
    error = None
    utility_error = None
    action = "predict"

    if request.method == "POST":
        action = request.form.get("action", "predict")
        threshold = sanitize_threshold(request.form.get("threshold", DEFAULT_THRESHOLD))
        variant_mode = request.form.get("variant_mode", DEFAULT_VARIANT_MODE)
        if variant_mode not in DESIGN_VARIANT_MODES:
            variant_mode = DEFAULT_VARIANT_MODE

        if action in UTILITY_ACTIONS:
            utility_raw_sequences = request.form.get("utility_sequences", "")
            uploaded = request.files.get("utility_file")
            if uploaded and uploaded.filename:
                uploaded_text = uploaded.read().decode("utf-8", errors="replace")
                utility_raw_sequences = f"{utility_raw_sequences.strip()}\n{uploaded_text.strip()}".strip()
            try:
                records = parse_peptide_input(utility_raw_sequences)
                if not records:
                    raise ValueError("Submit at least one peptide sequence.")
                utility_results = build_utility_results(records)
                utility_summary = summarize_utility_results(utility_results)
                if action == "utility_download":
                    csv_text = utility_results_to_csv(utility_results)
                    return Response(
                        csv_text,
                        mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=peptide_utilities.csv"},
                    )
                if action == "utility_report_pack":
                    zip_bytes = build_utility_report_pack(utility_results, utility_summary)
                    return Response(
                        zip_bytes,
                        mimetype="application/zip",
                        headers={"Content-Disposition": "attachment; filename=peptide_utilities_report_pack.zip"},
                    )
            except ValueError as exc:
                utility_error = str(exc)
        else:
            raw_sequences = request.form.get("sequences", "")
            uploaded = request.files.get("sequence_file")
            if uploaded and uploaded.filename:
                uploaded_text = uploaded.read().decode("utf-8", errors="replace")
                raw_sequences = f"{raw_sequences.strip()}\n{uploaded_text.strip()}".strip()
            try:
                records = parse_peptide_input(raw_sequences)
                if not records:
                    raise ValueError("Submit at least one peptide sequence.")
                results = build_prediction_results(records, threshold)
                summary = summarize_results(results)
                variant_parent = top_valid_result(results)

                if action in DESIGN_ACTIONS:
                    if not variant_parent and action in {"variant_download", "variant_fasta", "variant_report_pack"}:
                        raise ValueError("No valid parent sequence is available for variant export.")
                    if variant_parent:
                        variant_results, variant_position_summary = build_variant_results(
                            variant_parent,
                            variant_mode,
                            threshold,
                        )
                    if action in {"variant_download", "variant_fasta", "variant_report_pack"} and not variant_results:
                        raise ValueError("No variants could be generated for the selected parent and utility.")
                    if action == "variant_download":
                        csv_text = variant_results_to_csv(variant_parent, variant_results)
                        return Response(
                            csv_text,
                            mimetype="text/csv",
                            headers={"Content-Disposition": "attachment; filename=afpred_variants.csv"},
                        )
                    if action == "variant_fasta":
                        fasta_text = results_to_fasta(variant_results, include_variants=True)
                        return Response(
                            fasta_text,
                            mimetype="text/plain",
                            headers={"Content-Disposition": "attachment; filename=afpred_variants.fasta"},
                        )
                    if action == "variant_report_pack":
                        zip_bytes = build_report_pack(
                            results,
                            summary,
                            threshold,
                            variant_parent=variant_parent,
                            variant_results=variant_results,
                            position_summary=variant_position_summary,
                            variant_mode=variant_mode,
                        )
                        return Response(
                            zip_bytes,
                            mimetype="application/zip",
                            headers={"Content-Disposition": "attachment; filename=afpred_variant_report_pack.zip"},
                        )

                if action == "download":
                    csv_text = results_to_csv(results)
                    return Response(
                        csv_text,
                        mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=afpred_predictions.csv"},
                    )
                if action == "fasta_download":
                    fasta_text = results_to_fasta(results)
                    return Response(
                        fasta_text,
                        mimetype="text/plain",
                        headers={"Content-Disposition": "attachment; filename=afpred_predictions.fasta"},
                    )
                if action == "report_pack":
                    zip_bytes = build_report_pack(results, summary, threshold)
                    return Response(
                        zip_bytes,
                        mimetype="application/zip",
                        headers={"Content-Disposition": "attachment; filename=afpred_report_pack.zip"},
                    )
            except ValueError as exc:
                error = str(exc)

    if action in UTILITY_ACTIONS:
        active_tab = "utilities"
    elif action in DESIGN_ACTIONS:
        active_tab = "design"
    else:
        active_tab = "predict"

    return render_template(
        "index.html",
        error=error,
        utility_error=utility_error,
        max_length=MAX_LENGTH,
        min_length=MIN_LENGTH,
        raw_sequences=raw_sequences,
        utility_raw_sequences=utility_raw_sequences,
        results=results,
        utility_results=utility_results,
        summary=summary,
        utility_summary=utility_summary,
        threshold=threshold,
        variant_mode=variant_mode,
        variant_modes=DESIGN_VARIANT_MODES,
        variant_parent=variant_parent,
        variant_results=variant_results,
        variant_position_summary=variant_position_summary,
        action=action,
        active_tab=active_tab,
        app_version=APP_VERSION,
        model_name="AFPRED",
        lab_name="Laboratory of Bioactive Peptides",
        contact_email="sanchisivan@fbcb.unl.edu.ar",
        group_leader_email="asiano@fbcb.unl.edu.ar",
        external_model_candidates=EXTERNAL_MODEL_CANDIDATES,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    threshold = sanitize_threshold(payload.get("threshold", DEFAULT_THRESHOLD))

    if "sequences" in payload:
        raw_sequences = "\n".join(str(sequence) for sequence in payload["sequences"])
    else:
        raw_sequences = str(payload.get("sequence", ""))

    try:
        records = parse_peptide_input(raw_sequences)
        if not records:
            raise ValueError("Submit at least one peptide sequence.")
        results = build_prediction_results(records, threshold)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "threshold": threshold,
            "summary": summarize_results(results),
            "results": results,
        }
    )


@app.route("/api/properties", methods=["POST"])
def api_properties():
    payload = request.get_json(silent=True) or {}

    if "sequences" in payload:
        raw_sequences = "\n".join(str(sequence) for sequence in payload["sequences"])
    else:
        raw_sequences = str(payload.get("sequence", ""))

    try:
        records = parse_peptide_input(raw_sequences)
        if not records:
            raise ValueError("Submit at least one peptide sequence.")
        results = build_utility_results(records)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "summary": summarize_utility_results(results),
            "results": results,
        }
    )


@app.route("/api/variants", methods=["POST"])
def api_variants():
    payload = request.get_json(silent=True) or {}
    threshold = sanitize_threshold(payload.get("threshold", DEFAULT_THRESHOLD))
    mode = payload.get("mode", "activity_optimization")
    if mode not in VARIANT_MODES:
        return jsonify({"error": f"Unsupported variant mode: {mode}"}), 400
    if mode == "none":
        return jsonify({"error": "Choose a variant mode other than none."}), 400

    try:
        sequence = validate_sequence(str(payload.get("sequence", "")))
        parent = {
            "id": "parent",
            "sequence": sequence,
            "probability": None,
        }
        parent_result = build_prediction_results([parent], threshold)[0]
        variants, position_summary = build_variant_results(parent_result, mode, threshold)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "threshold": threshold,
            "mode": mode,
            "mode_label": VARIANT_MODES[mode],
            "parent": parent_result,
            "position_summary": position_summary,
            "variants": variants,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "version": APP_VERSION,
            "model_path": MODEL_PATH,
            "model_available": os.path.exists(MODEL_PATH),
            "sequence_length_range": [MIN_LENGTH, MAX_LENGTH],
            "variant_modes": DESIGN_VARIANT_MODES,
        }
    )


@app.route("/version", methods=["GET"])
def version():
    return jsonify(
        {
            "app": "AFPRED",
            "version": APP_VERSION,
            "reports_export_enabled": True,
            "report_actions": [
                "report_pack",
                "variant_report_pack",
                "utility_report_pack",
            ],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
