import csv
import io
import json
import os
from functools import lru_cache

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
DEFAULT_VARIANT_MODE = "activity_optimization"
DESIGN_ACTIONS = {"variants", "variant_download", "variant_fasta"}
UTILITY_ACTIONS = {"utilities", "utility_download"}
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
                    if not variant_parent and action in {"variant_download", "variant_fasta"}:
                        raise ValueError("No valid parent sequence is available for variant export.")
                    if variant_parent:
                        variant_results, variant_position_summary = build_variant_results(
                            variant_parent,
                            variant_mode,
                            threshold,
                        )
                    if action in {"variant_download", "variant_fasta"} and not variant_results:
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
            "model_path": MODEL_PATH,
            "model_available": os.path.exists(MODEL_PATH),
            "sequence_length_range": [MIN_LENGTH, MAX_LENGTH],
            "variant_modes": DESIGN_VARIANT_MODES,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
