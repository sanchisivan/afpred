import csv
import re
from math import log2, pi

import numpy as np


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa.lower(): index + 1 for index, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}
VALID_AA = set(AMINO_ACIDS)
MAX_LENGTH = 40
MIN_LENGTH = 6
MAX_BATCH_SIZE = 100
DEFAULT_THRESHOLD = 0.8
MAX_VARIANTS = 40

RESIDUE_MASS = {
    "A": 71.0788,
    "R": 156.1875,
    "N": 114.1038,
    "D": 115.0886,
    "C": 103.1388,
    "Q": 128.1307,
    "E": 129.1155,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "L": 113.1594,
    "K": 128.1741,
    "M": 131.1926,
    "F": 147.1766,
    "P": 97.1167,
    "S": 87.0782,
    "T": 101.1051,
    "W": 186.2132,
    "Y": 163.1760,
    "V": 99.1326,
}

RESIDUE_FORMULA = {
    "A": {"C": 3, "H": 5, "N": 1, "O": 1, "S": 0},
    "R": {"C": 6, "H": 12, "N": 4, "O": 1, "S": 0},
    "N": {"C": 4, "H": 6, "N": 2, "O": 2, "S": 0},
    "D": {"C": 4, "H": 5, "N": 1, "O": 3, "S": 0},
    "C": {"C": 3, "H": 5, "N": 1, "O": 1, "S": 1},
    "Q": {"C": 5, "H": 8, "N": 2, "O": 2, "S": 0},
    "E": {"C": 5, "H": 7, "N": 1, "O": 3, "S": 0},
    "G": {"C": 2, "H": 3, "N": 1, "O": 1, "S": 0},
    "H": {"C": 6, "H": 7, "N": 3, "O": 1, "S": 0},
    "I": {"C": 6, "H": 11, "N": 1, "O": 1, "S": 0},
    "L": {"C": 6, "H": 11, "N": 1, "O": 1, "S": 0},
    "K": {"C": 6, "H": 12, "N": 2, "O": 1, "S": 0},
    "M": {"C": 5, "H": 9, "N": 1, "O": 1, "S": 1},
    "F": {"C": 9, "H": 9, "N": 1, "O": 1, "S": 0},
    "P": {"C": 5, "H": 7, "N": 1, "O": 1, "S": 0},
    "S": {"C": 3, "H": 5, "N": 1, "O": 2, "S": 0},
    "T": {"C": 4, "H": 7, "N": 1, "O": 2, "S": 0},
    "W": {"C": 11, "H": 10, "N": 2, "O": 1, "S": 0},
    "Y": {"C": 9, "H": 9, "N": 1, "O": 2, "S": 0},
    "V": {"C": 5, "H": 9, "N": 1, "O": 1, "S": 0},
}

KYTE_DOOLITTLE = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

EISENBERG_CONSENSUS = {
    "A": 0.62,
    "R": -2.53,
    "N": -0.78,
    "D": -0.90,
    "C": 0.29,
    "Q": -0.85,
    "E": -0.74,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "L": 1.06,
    "K": -1.50,
    "M": 0.64,
    "F": 1.19,
    "P": 0.12,
    "S": -0.18,
    "T": -0.05,
    "W": 0.81,
    "Y": 0.26,
    "V": 1.08,
}

BOMAN_TRANSFER_FREE_ENERGY = {
    "A": 1.81,
    "R": -14.92,
    "N": -6.64,
    "D": -8.72,
    "C": 1.28,
    "Q": -5.54,
    "E": -6.81,
    "G": 0.94,
    "H": -4.66,
    "I": 4.92,
    "L": 4.92,
    "K": -5.55,
    "M": 2.35,
    "F": 2.98,
    "P": 0.0,
    "S": -3.4,
    "T": -2.57,
    "W": 2.33,
    "Y": -0.14,
    "V": 4.04,
}

HYDROPHOBIC_AA = set("AVILMFWYC")
AROMATIC_AA = set("FWY")
POSITIVE_AA = set("KRH")
NEGATIVE_AA = set("DE")
POLAR_AA = set("STNQ")
SPECIAL_AA = set("GP")
DISORDER_PROMOTING_AA = set("ARGQSEKP")
BETA_BRANCHED_AROMATIC_AA = set("VIFYWT")
AMBIGUOUS_OR_UNSUPPORTED = {
    "B": "aspartate/asparagine ambiguity",
    "J": "leucine/isoleucine ambiguity",
    "O": "pyrrolysine is not supported by this model",
    "U": "selenocysteine is not supported by this model",
    "X": "unknown residue",
    "Z": "glutamate/glutamine ambiguity",
}
ID_HEADER_NAMES = {"id", "name", "label", "record", "sample", "candidate", "peptide_id"}
SEQUENCE_HEADER_NAMES = {"sequence", "seq", "peptide", "peptide_sequence", "aa_sequence"}
RESIDUE_GROUPS = {
    "positive": POSITIVE_AA,
    "negative": NEGATIVE_AA,
    "polar": POLAR_AA,
    "hydrophobic": HYDROPHOBIC_AA,
    "aromatic": AROMATIC_AA,
    "special": SPECIAL_AA,
}
PKA_VALUES = {
    "Cterm": 3.55,
    "Nterm": 7.5,
    "C": 8.5,
    "D": 3.9,
    "E": 4.1,
    "H": 6.5,
    "K": 10.8,
    "R": 12.5,
    "Y": 10.1,
}


def clean_sequence(sequence):
    return re.sub(r"[\s\-]", "", sequence or "").upper()


def input_diagnostics(sequence):
    raw = sequence or ""
    cleaned = clean_sequence(raw)
    raw_nonspace = re.sub(r"\s", "", raw)
    invalid = sorted(set(cleaned) - VALID_AA)
    unsupported = [
        {"residue": aa, "reason": AMBIGUOUS_OR_UNSUPPORTED.get(aa, "non-canonical residue")}
        for aa in invalid
    ]
    notices = []

    if raw != cleaned:
        notices.append("Whitespace and hyphens are ignored before validation.")
    if re.search(r"(^|[^A-Za-z])(ac|nh2|amide|acetyl)([^A-Za-z]|$)", raw, flags=re.IGNORECASE):
        notices.append("Terminal modifications are not represented in the model input.")
    if re.search(r"[^A-Za-z\s>\-;,_]", raw):
        notices.append("Symbols or punctuation were detected and are not part of canonical peptide notation.")

    return {
        "raw_length": len(raw_nonspace),
        "cleaned_length": len(cleaned),
        "cleaned_sequence": cleaned,
        "invalid_residues": unsupported,
        "notices": notices,
    }


def validate_sequence(sequence):
    cleaned = clean_sequence(sequence)

    if not cleaned:
        raise ValueError("Empty sequence.")

    errors = []
    invalid = sorted(set(cleaned) - VALID_AA)
    if invalid:
        invalid_text = ", ".join(invalid)
        errors.append(f"invalid residue(s): {invalid_text}")
    if not (MIN_LENGTH <= len(cleaned) <= MAX_LENGTH):
        errors.append(f"length must be between {MIN_LENGTH} and {MAX_LENGTH} amino acids")
    if errors:
        raise ValueError(f"Sequence rejected: {'; '.join(errors)}.")

    return cleaned


def header_key(value):
    return re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")


def plain_sequence_token(value):
    raw = (value or "").strip()
    if not raw or re.search(r"[^A-Za-z\s\-]", raw):
        return False
    cleaned = clean_sequence(raw)
    return bool(cleaned) and set(cleaned).issubset(VALID_AA)


def delimited_cells(line):
    delimiter = None
    if "\t" in line:
        delimiter = "\t"
    elif "," in line:
        delimiter = ","
    elif ";" in line:
        delimiter = ";"
    if delimiter is None:
        return None
    return next(csv.reader([line], delimiter=delimiter, skipinitialspace=True))


def append_record(records, sequence, record_id=None):
    record_id = (record_id or "").strip() or f"peptide_{len(records) + 1}"
    records.append(
        {
            "id": record_id,
            "sequence": (sequence or "").strip(),
        }
    )


def encode_sequence(sequence):
    cleaned = validate_sequence(sequence)
    encoded = [AA_TO_INT[aa.lower()] for aa in cleaned]
    padded = np.zeros((1, MAX_LENGTH), dtype=np.int32)
    padded[0, : len(encoded)] = encoded
    return padded


def parse_peptide_input(raw_text):
    text = (raw_text or "").strip()
    if not text:
        return []

    records = []
    if ">" in text:
        current_id = None
        current_lines = []

        def flush_record():
            if current_id is not None:
                records.append(
                    {
                        "id": current_id,
                        "sequence": "".join(current_lines),
                    }
                )

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                current_id = line[1:].strip() or f"peptide_{len(records) + 1}"
                current_lines = []
            else:
                current_lines.append(line)
        flush_record()
    else:
        header = None

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            cells = delimited_cells(line)
            if cells is None:
                append_record(records, line)
                continue

            cells = [cell.strip() for cell in cells if cell.strip()]
            if not cells:
                continue

            normalized = [header_key(cell) for cell in cells]
            if any(name in SEQUENCE_HEADER_NAMES for name in normalized):
                header = normalized
                continue

            sequence_index = None
            id_index = None
            if header:
                for index, column in enumerate(header[: len(cells)]):
                    if column in SEQUENCE_HEADER_NAMES and sequence_index is None:
                        sequence_index = index
                    if column in ID_HEADER_NAMES and id_index is None:
                        id_index = index

            if sequence_index is not None and sequence_index < len(cells):
                record_id = cells[id_index] if id_index is not None and id_index < len(cells) else None
                append_record(records, cells[sequence_index], record_id)
                continue

            sequence_indexes = [index for index, cell in enumerate(cells) if plain_sequence_token(cell)]
            all_sequence_cells = len(sequence_indexes) == len(cells)
            first_cell_is_identifier = bool(re.search(r"[^A-Za-z\s\-]", cells[0]))

            if len(cells) > 1 and all_sequence_cells:
                for cell in cells:
                    append_record(records, cell)
            elif len(cells) > 1 and sequence_indexes:
                preferred = next((index for index in sequence_indexes if index > 0), sequence_indexes[0])
                record_id = cells[0] if first_cell_is_identifier or preferred > 0 else None
                append_record(records, cells[preferred], record_id)
            else:
                for cell in cells:
                    append_record(records, cell)

    if len(records) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch limit exceeded. Submit at most {MAX_BATCH_SIZE} peptides at a time.")

    return records


def shannon_entropy(counts, length):
    entropy = 0.0
    for count in counts.values():
        if count:
            p = count / length
            entropy -= p * log2(p)
    return entropy / log2(len(AMINO_ACIDS))


def charge_at_ph(sequence, ph):
    counts = {aa: sequence.count(aa) for aa in AMINO_ACIDS}
    positive = (
        1 / (1 + 10 ** (ph - PKA_VALUES["Nterm"]))
        + counts["K"] / (1 + 10 ** (ph - PKA_VALUES["K"]))
        + counts["R"] / (1 + 10 ** (ph - PKA_VALUES["R"]))
        + counts["H"] / (1 + 10 ** (ph - PKA_VALUES["H"]))
    )
    negative = (
        1 / (1 + 10 ** (PKA_VALUES["Cterm"] - ph))
        + counts["D"] / (1 + 10 ** (PKA_VALUES["D"] - ph))
        + counts["E"] / (1 + 10 ** (PKA_VALUES["E"] - ph))
        + counts["C"] / (1 + 10 ** (PKA_VALUES["C"] - ph))
        + counts["Y"] / (1 + 10 ** (PKA_VALUES["Y"] - ph))
    )
    return positive - negative


def estimate_pi(sequence):
    low = 0.0
    high = 14.0
    for _ in range(45):
        mid = (low + high) / 2
        if charge_at_ph(sequence, mid) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def scale_moment(sequence, scale, angle_degrees=100):
    angle = angle_degrees * pi / 180
    x_component = 0.0
    y_component = 0.0
    for index, aa in enumerate(sequence):
        hydropathy = scale[aa]
        theta = index * angle
        x_component += hydropathy * np.cos(theta)
        y_component += hydropathy * np.sin(theta)
    return float((x_component**2 + y_component**2) ** 0.5 / len(sequence))


def hydrophobic_moment(sequence, angle_degrees=100):
    return scale_moment(sequence, KYTE_DOOLITTLE, angle_degrees)


def mean_scale(sequence, scale):
    return sum(scale[aa] for aa in sequence) / len(sequence)


def residue_composition(counts, length):
    rows = []
    for aa in AMINO_ACIDS:
        group = "other"
        for group_name, residues in RESIDUE_GROUPS.items():
            if aa in residues:
                group = group_name
                break
        rows.append(
            {
                "aa": aa,
                "count": counts[aa],
                "percent": round(100 * counts[aa] / length, 1),
                "group": group,
            }
        )
    return rows


def group_percent(counts, residues, length):
    return round(100 * sum(counts[aa] for aa in residues) / length, 1)


def molecular_formula(sequence):
    totals = {"C": 0, "H": 2, "N": 0, "O": 1, "S": 0}
    for aa in sequence:
        for element, count in RESIDUE_FORMULA[aa].items():
            totals[element] += count
    parts = []
    for element in ["C", "H", "N", "O", "S"]:
        count = totals[element]
        if count:
            parts.append(element if count == 1 else f"{element}{count}")
    return "".join(parts)


def extinction_coefficients(sequence, molecular_weight):
    trp = sequence.count("W")
    tyr = sequence.count("Y")
    cysteine = sequence.count("C")
    oxidized = 5500 * trp + 1490 * tyr + 125 * (cysteine // 2)
    reduced = 5500 * trp + 1490 * tyr
    return {
        "extinction_reduced": oxidized if cysteine < 2 else reduced,
        "extinction_oxidized": oxidized,
        "absorbance_0_1_percent_reduced": round(reduced / molecular_weight, 3) if molecular_weight else 0,
        "absorbance_0_1_percent_oxidized": round(oxidized / molecular_weight, 3) if molecular_weight else 0,
    }


def aliphatic_index(counts, length):
    ala = 100 * counts["A"] / length
    val = 100 * counts["V"] / length
    ile = 100 * counts["I"] / length
    leu = 100 * counts["L"] / length
    return ala + 2.9 * val + 3.9 * (ile + leu)


def terminal_profile(sequence):
    window = min(5, len(sequence))
    n_term = sequence[:window]
    c_term = sequence[-window:]
    return {
        "n_terminal": n_term,
        "c_terminal": c_term,
        "window_size": window,
        "windows_overlap": len(sequence) < window * 2,
        "n_terminal_charge": round(charge_at_ph(n_term, 7.0), 2),
        "c_terminal_charge": round(charge_at_ph(c_term, 7.0), 2),
        "n_terminal_hydrophobic_percent": group_percent({aa: n_term.count(aa) for aa in AMINO_ACIDS}, HYDROPHOBIC_AA, len(n_term)),
        "c_terminal_hydrophobic_percent": group_percent({aa: c_term.count(aa) for aa in AMINO_ACIDS}, HYDROPHOBIC_AA, len(c_term)),
    }


def sliding_window_profiles(sequence, window=5):
    if len(sequence) < window:
        window = len(sequence)
    rows = []
    for start in range(0, len(sequence) - window + 1):
        fragment = sequence[start : start + window]
        rows.append(
            {
                "start": start + 1,
                "end": start + window,
                "fragment": fragment,
                "charge": round(charge_at_ph(fragment, 7.0), 2),
                "hydropathy": round(sum(KYTE_DOOLITTLE[aa] for aa in fragment) / window, 3),
                "hydrophobic_percent": group_percent({aa: fragment.count(aa) for aa in AMINO_ACIDS}, HYDROPHOBIC_AA, window),
            }
        )
    return rows


def longest_run(sequence, residues):
    best = 0
    current = 0
    for aa in sequence:
        if aa in residues:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def boman_index(sequence):
    return -sum(BOMAN_TRANSFER_FREE_ENERGY[aa] for aa in sequence) / len(sequence)


def hydrophobic_face_angle(sequence, angle_degrees=100):
    angles = sorted((index * angle_degrees) % 360 for index, aa in enumerate(sequence) if aa in HYDROPHOBIC_AA)
    if not angles:
        return 0
    if len(angles) == 1:
        return 0
    gaps = []
    for index, angle in enumerate(angles):
        next_angle = angles[(index + 1) % len(angles)]
        gap = (next_angle - angle) % 360
        gaps.append(gap)
    return 360 - max(gaps)


def linear_hydrophobic_moment(sequence, scale=KYTE_DOOLITTLE):
    if len(sequence) == 1:
        return 0
    center = (len(sequence) - 1) / 2
    numerator = sum((index - center) * scale[aa] for index, aa in enumerate(sequence))
    denominator = len(sequence) * center * max(abs(value) for value in scale.values())
    return numerator / denominator if denominator else 0


def charge_profile(sequence):
    return [
        {"ph": ph, "charge": round(charge_at_ph(sequence, ph), 2)}
        for ph in [3.0, 5.5, 7.0, 7.4, 9.0, 11.0]
    ]


def hydrophobic_hotspot(sliding_windows):
    if not sliding_windows:
        return {}
    return max(sliding_windows, key=lambda row: (row["hydropathy"], row["hydrophobic_percent"]))


def amp_property_panel(sequence, counts, length, gravy, hydrophobic_percent, h_moment, entropy):
    acidic = counts["D"] + counts["E"]
    basic = counts["K"] + counts["R"] + counts["H"]
    hydrophobic_run = longest_run(sequence, HYDROPHOBIC_AA)
    beta_aggregation_run = longest_run(sequence, BETA_BRANCHED_AROMATIC_AA)
    disorder_percent = group_percent(counts, DISORDER_PROMOTING_AA, length)
    basic_to_acidic = None if acidic == 0 else round(basic / acidic, 2)
    aggregation_score = 0
    if hydrophobic_percent >= 55:
        aggregation_score += 2
    if hydrophobic_run >= 5:
        aggregation_score += 2
    if beta_aggregation_run >= 4:
        aggregation_score += 1
    if entropy < 0.5:
        aggregation_score += 1

    eisenberg_hydrophobicity = float(mean_scale(sequence, EISENBERG_CONSENSUS))
    eisenberg_moment = float(scale_moment(sequence, EISENBERG_CONSENSUS))
    helical_discriminant = float(0.944 * eisenberg_moment + 0.33 * charge_at_ph(sequence, 7.4))

    return {
        "kd_hydrophobicity_mean": round(gravy, 3),
        "normalized_hydrophobic_moment_kd": round(h_moment / max(abs(value) for value in KYTE_DOOLITTLE.values()), 3),
        "eisenberg_hydrophobicity": round(eisenberg_hydrophobicity, 3),
        "eisenberg_hydrophobic_moment": round(eisenberg_moment, 3),
        "heliquest_like_discriminant": round(helical_discriminant, 3),
        "heliquest_like_lipid_binding": bool(helical_discriminant > 0.68),
        "heliquest_like_transmembrane": bool(eisenberg_hydrophobicity > 0.75),
        "hydrophobic_face_angle": round(hydrophobic_face_angle(sequence), 1),
        "linear_hydrophobic_moment": round(linear_hydrophobic_moment(sequence), 3),
        "linear_moment_eisenberg": round(linear_hydrophobic_moment(sequence, EISENBERG_CONSENSUS), 3),
        "boman_index": round(boman_index(sequence), 3),
        "disorder_promoting_percent": disorder_percent,
        "max_hydrophobic_run": hydrophobic_run,
        "max_beta_aggregation_run": beta_aggregation_run,
        "basic_residue_count": basic,
        "acidic_residue_count": acidic,
        "basic_to_acidic_ratio": basic_to_acidic,
        "aggregation_risk_score": aggregation_score,
    }


def afp_feature_score(properties):
    score = 0
    if properties["net_charge"] >= 2:
        score += 2
    if properties["hydrophobic_moment"] >= 0.35:
        score += 2
    if 25 <= properties["hydrophobic_percent"] <= 55:
        score += 1
    if properties["sequence_entropy"] >= 0.55:
        score += 1
    if properties["cysteines"] >= 2:
        score += 1
    if properties["isoelectric_point"] >= 8:
        score += 1
    return min(score, 8)


def screening_recommendation(probability, properties):
    afp_score = afp_feature_score(properties)
    liabilities = []
    if properties["hydrophobic_percent"] >= 60:
        liabilities.append("hydrophobicity")
    if properties["sequence_entropy"] < 0.45:
        liabilities.append("low complexity")
    if properties["net_charge"] >= 8:
        liabilities.append("very high charge")
    if properties["length"] < 8 or properties["length"] > 35:
        liabilities.append("edge length")

    if probability is None:
        tier = "not ranked"
    elif probability >= 0.8 and afp_score >= 5 and len(liabilities) <= 1:
        tier = "advance"
    elif probability >= 0.5 or afp_score >= 4:
        tier = "review"
    else:
        tier = "deprioritize"

    return {
        "tier": tier,
        "afp_feature_score": afp_score,
        "liabilities": ", ".join(liabilities) if liabilities else "none flagged",
    }


def chemical_liabilities(sequence):
    liabilities = []
    counts = {aa: sequence.count(aa) for aa in AMINO_ACIDS}
    length = len(sequence)
    hydrophobic_percent = group_percent(counts, HYDROPHOBIC_AA, length)
    aromatic_percent = group_percent(counts, AROMATIC_AA, length)
    polar_percent = group_percent(counts, POLAR_AA, length)
    entropy = shannon_entropy(counts, length)
    charge_7 = charge_at_ph(sequence, 7.0)
    pi_value = estimate_pi(sequence)
    hydrophobic_run = longest_run(sequence, HYDROPHOBIC_AA)
    beta_run = longest_run(sequence, BETA_BRANCHED_AROMATIC_AA)

    def add(kind, title, detail):
        liabilities.append({"kind": kind, "title": title, "detail": detail})

    def motif_positions(pattern):
        return [match.start() + 1 for match in re.finditer(f"(?=({pattern}))", sequence)]

    if counts["M"]:
        add("caution", "Methionine oxidation risk", "Met can oxidize to sulfoxide/sulfone during storage, handling, or oxidative biological exposure.")
    if counts["W"]:
        add("caution", "Tryptophan oxidation/photooxidation risk", "Trp can be sensitive to light and oxidants; confirm identity/purity after stress or long storage.")
    if counts["C"] == 1:
        add("caution", "Single free cysteine", "A lone Cys can form disulfide-linked dimers or thiol adducts unless protected or intentionally modified.")
    elif counts["C"] > 1:
        add("caution", "Cysteine/disulfide ambiguity", "Multiple Cys residues require explicit control of reduced, oxidized, or disulfide-paired state.")
    if counts["Y"] >= 2:
        add("context", "Tyrosine oxidation/crosslinking context", "Multiple Tyr residues can be oxidation-sensitive under some stress conditions.")

    asn_positions = motif_positions(r"N[GST]")
    if asn_positions:
        add("caution", "Asn deamidation motif", f"N-G/S/T motif detected near position(s) {', '.join(map(str, asn_positions))}; deamidation risk depends on pH, temperature, buffer, and neighboring residues.")
    gln_positions = motif_positions(r"Q[GST]")
    if gln_positions:
        add("context", "Gln deamidation motif", f"Q-G/S/T motif detected near position(s) {', '.join(map(str, gln_positions))}; usually slower than Asn but worth tracking in stability studies.")
    asp_iso_positions = motif_positions(r"D[GST]")
    if asp_iso_positions:
        add("caution", "Asp isomerization/hydrolysis motif", f"D-G/S/T motif detected near position(s) {', '.join(map(str, asp_iso_positions))}; Asp-Gly/Ser/Thr motifs can be stability liabilities.")
    if re.search(r"DP", sequence):
        add("caution", "Asp-Pro liability", "D-P motifs can be acid-labile during handling or analytical workflows.")
    if sequence.startswith("Q"):
        add("caution", "N-terminal Gln cyclization risk", "N-terminal Gln can cyclize to pyroglutamate, changing mass and charge.")
    if sequence.startswith("E"):
        add("context", "N-terminal Glu cyclization context", "N-terminal Glu can form pyroglutamate more slowly under some conditions.")
    if sequence.startswith("C"):
        add("context", "N-terminal cysteine reactivity", "N-terminal Cys can form adducts with aldehydes or participate in ligation-like chemistry.")

    trypsin_sites = [index + 1 for index, aa in enumerate(sequence[:-1]) if aa in "KR" and sequence[index + 1] != "P"]
    if trypsin_sites:
        add("context", "Trypsin-like cleavage sites", f"K/R sites not followed by Pro at position(s) {', '.join(map(str, trypsin_sites))}; relevant for protease-exposure or serum-stability planning.")
    chymo_sites = [index + 1 for index, aa in enumerate(sequence[:-1]) if aa in "FYW" and sequence[index + 1] != "P"]
    if chymo_sites:
        add("context", "Chymotrypsin-like cleavage sites", f"F/Y/W sites not followed by Pro at position(s) {', '.join(map(str, chymo_sites))}; relevant for enzymatic-stability follow-up.")

    if hydrophobic_percent >= 55:
        add("caution", "High hydrophobic fraction", "High hydrophobic content can reduce aqueous solubility and increase nonspecific membrane binding or hemolysis risk.")
    if hydrophobic_run >= 5:
        add("caution", "Long hydrophobic stretch", f"Longest hydrophobic run is {hydrophobic_run} residues; this can complicate synthesis, purification, and solubility.")
    if beta_run >= 4:
        add("caution", "Beta/aggregation-prone stretch", f"Longest V/I/F/Y/W/T-rich run is {beta_run} residues; inspect aggregation risk experimentally.")
    if aromatic_percent >= 25:
        add("context", "Aromatic enrichment", "High aromatic content can support membrane binding but may increase nonspecific binding, oxidation sensitivity, or aggregation.")
    if polar_percent <= 15 and hydrophobic_percent >= 40:
        add("caution", "Low polar buffering", "Low polar content with moderate/high hydrophobicity can make solubility sensitive to salts, pH, and formulation.")
    if abs(charge_7) <= 0.5 and hydrophobic_percent >= 40:
        add("context", "Near-neutral hydrophobic peptide", "Near-neutral charge at pH 7 with hydrophobic content can increase precipitation/self-association risk.")
    if 6.0 <= pi_value <= 8.0 and abs(charge_7) <= 0.5:
        add("context", "pI near neutral pH", "The estimated pI is close to neutral pH, where solubility can be reduced for some peptides.")
    if abs(charge_7) >= 6:
        add("context", "Very high charge density", "Strong charge can improve solubility but may increase salt sensitivity, nonspecific binding, or assay interference.")
    if entropy < 0.45:
        add("context", "Low sequence complexity", "Low-complexity sequences can be harder to interpret and may bias sequence-only models.")
    if not liabilities:
        add("signal", "No major simple liabilities", "No common sequence-level chemical liability pattern was detected by this lightweight screen.")

    return liabilities


def residue_annotations(sequence):
    annotated = []
    for index, aa in enumerate(sequence, start=1):
        if aa in POSITIVE_AA:
            group = "positive"
            label = "basic/cationic"
        elif aa in NEGATIVE_AA:
            group = "negative"
            label = "acidic/anionic"
        elif aa in HYDROPHOBIC_AA:
            group = "hydrophobic"
            label = "hydrophobic"
        elif aa in AROMATIC_AA:
            group = "aromatic"
            label = "aromatic"
        elif aa in POLAR_AA:
            group = "polar"
            label = "polar"
        elif aa in SPECIAL_AA:
            group = "special"
            label = "turn/flexibility"
        else:
            group = "other"
            label = "other"
        annotated.append(
            {
                "position": index,
                "aa": aa,
                "group": group,
                "label": label,
                "hydropathy": KYTE_DOOLITTLE[aa],
            }
        )
    return annotated


def replace_at(sequence, index, residue):
    return sequence[:index] + residue + sequence[index + 1 :]


def generate_variants(sequence, mode="alanine_scan", max_variants=MAX_VARIANTS):
    seq = validate_sequence(sequence)
    variants = []
    seen = {seq}

    def add_variant(variant_id, variant_sequence, change, rationale):
        if len(variants) >= max_variants:
            return
        if variant_sequence in seen:
            return
        try:
            validate_sequence(variant_sequence)
        except ValueError:
            return
        seen.add(variant_sequence)
        variants.append(
            {
                "id": variant_id,
                "sequence": variant_sequence,
                "change": change,
                "rationale": rationale,
            }
        )

    if mode == "alanine_scan":
        for index, aa in enumerate(seq):
            if aa != "A":
                add_variant(
                    f"Ala_{index + 1}_{aa}A",
                    replace_at(seq, index, "A"),
                    f"{aa}{index + 1}A",
                    "Single-position alanine scan to test side-chain contribution.",
                )
    elif mode == "tryptophan_scan":
        for index, aa in enumerate(seq):
            if aa != "W":
                add_variant(
                    f"Trp_{index + 1}_{aa}W",
                    replace_at(seq, index, "W"),
                    f"{aa}{index + 1}W",
                    "Explores aromatic anchoring, inspired by Trp-enrichment strategies.",
                )
    elif mode == "lysine_scan":
        for index, aa in enumerate(seq):
            if aa != "K":
                add_variant(
                    f"Lys_{index + 1}_{aa}K",
                    replace_at(seq, index, "K"),
                    f"{aa}{index + 1}K",
                    "Increases cationic character at one position.",
                )
    elif mode == "hydrophobic_tempering":
        for index, aa in enumerate(seq):
            if aa in set("FWYILVM"):
                add_variant(
                    f"Temper_{index + 1}_{aa}A",
                    replace_at(seq, index, "A"),
                    f"{aa}{index + 1}A",
                    "Reduces hydrophobic/aromatic load to inspect selectivity risk.",
                )
    elif mode == "terminal_truncation":
        for trim in range(1, min(5, len(seq) - MIN_LENGTH + 1)):
            add_variant(
                f"Ntrim_{trim}",
                seq[trim:],
                f"N-terminal trim {trim}",
                "Tests whether the N-terminal segment is required for predicted signal.",
            )
            add_variant(
                f"Ctrim_{trim}",
                seq[:-trim],
                f"C-terminal trim {trim}",
                "Tests whether the C-terminal segment is required for predicted signal.",
            )

    return variants


def generate_substitution_library(sequence):
    seq = validate_sequence(sequence)
    variants = []
    for index, original in enumerate(seq):
        for residue in AMINO_ACIDS:
            if residue == original:
                continue
            variants.append(
                {
                    "id": f"Sub_{index + 1}_{original}{residue}",
                    "sequence": replace_at(seq, index, residue),
                    "change": f"{original}{index + 1}{residue}",
                    "position": index + 1,
                    "from": original,
                    "to": residue,
                    "rationale": "Single-residue substitution screened for predicted activity gain at this position.",
                }
            )
    return variants


def design_alerts(sequence, properties):
    alerts = []

    def add(kind, title, detail):
        alerts.append({"kind": kind, "title": title, "detail": detail})

    if properties["net_charge"] >= 3 and properties["hydrophobic_moment"] >= 0.35:
        add("signal", "Membrane-active profile", "Cationic charge plus amphipathic signal is consistent with many AFP scaffolds.")
    if properties["hydrophobic_percent"] >= 55:
        add("caution", "High hydrophobic fraction", "Prioritize solubility and hemolysis checks during follow-up.")
    if properties["net_charge"] <= 0:
        add("caution", "Low cationic character", "Many short AFPs are cationic; this sequence may be outside the common membrane-active profile.")
    if properties["aromaticity"] >= 0.2:
        add("signal", "Aromatic-rich sequence", "Aromatic residues can support membrane anchoring but may also increase nonspecific binding.")
    if properties["cysteines"] >= 2:
        add("context", "Cysteine-rich candidate", "Consider whether disulfide bonding or cyclization is intended; the current model receives only linear sequence text.")
    if re.search(r"[KR]{4,}", sequence):
        add("context", "Basic residue cluster", "A long Lys/Arg run can affect selectivity, solubility, and uptake behavior.")
    if properties["sequence_entropy"] < 0.55:
        add("caution", "Low sequence complexity", "Low-complexity sequences can be model-sensitive and should be inspected manually.")
    if not alerts:
        add("context", "No major descriptor alerts", "Use the model score together with external toxicity and novelty checks.")

    return alerts


def peptide_properties(sequence):
    seq = validate_sequence(sequence)
    length = len(seq)
    counts = {aa: seq.count(aa) for aa in AMINO_ACIDS}
    mass = sum(RESIDUE_MASS[aa] for aa in seq) + 18.01528
    net_charge = counts["K"] + counts["R"] + 0.1 * counts["H"] - counts["D"] - counts["E"]
    gravy = sum(KYTE_DOOLITTLE[aa] for aa in seq) / length
    hydrophobic_percent = 100 * sum(counts[aa] for aa in HYDROPHOBIC_AA) / length
    aromaticity = sum(counts[aa] for aa in AROMATIC_AA) / length
    entropy = shannon_entropy(counts, length)
    h_moment = hydrophobic_moment(seq)

    notes = []
    if net_charge >= 2:
        notes.append("cationic")
    if h_moment >= 0.35:
        notes.append("amphipathic")
    if gravy >= 1.0:
        notes.append("hydrophobic")
    if counts["C"] >= 2:
        notes.append("cysteine-rich")
    if counts["P"] >= 3:
        notes.append("proline-rich")
    if not notes:
        notes.append("balanced")

    properties = {
        "length": length,
        "formula": molecular_formula(seq),
        "molecular_weight": round(mass, 2),
        "net_charge": round(net_charge, 2),
        "charge_ph_5_5": round(charge_at_ph(seq, 5.5), 2),
        "charge_ph_7_0": round(charge_at_ph(seq, 7.0), 2),
        "charge_ph_7_4": round(charge_at_ph(seq, 7.4), 2),
        "charge_density": round(net_charge / length, 3),
        "gravy": round(gravy, 3),
        "aliphatic_index": round(aliphatic_index(counts, length), 2),
        "hydrophobic_percent": round(hydrophobic_percent, 1),
        "positive_percent": group_percent(counts, POSITIVE_AA, length),
        "negative_percent": group_percent(counts, NEGATIVE_AA, length),
        "polar_percent": group_percent(counts, POLAR_AA, length),
        "aromaticity": round(aromaticity, 3),
        "hydrophobic_moment": round(h_moment, 3),
        "isoelectric_point": round(estimate_pi(seq), 2),
        "sequence_entropy": round(entropy, 3),
        "cysteines": counts["C"],
        "notes": ", ".join(notes),
        "composition": residue_composition(counts, length),
        "residue_annotations": residue_annotations(seq),
        "terminal_profile": terminal_profile(seq),
        "chemical_liabilities": chemical_liabilities(seq),
    }
    properties["sliding_windows"] = sliding_window_profiles(seq)
    properties["hydrophobic_hotspot"] = hydrophobic_hotspot(properties["sliding_windows"])
    properties["charge_profile"] = charge_profile(seq)
    properties.update(amp_property_panel(seq, counts, length, gravy, hydrophobic_percent, h_moment, entropy))
    properties.update(extinction_coefficients(seq, mass))
    properties["afp_feature_score"] = afp_feature_score(properties)
    properties["alerts"] = design_alerts(seq, properties)
    return properties


def classify_probability(probability, threshold=DEFAULT_THRESHOLD):
    if probability >= threshold:
        return {
            "classification": "Likely antifungal",
            "status": "high",
            "alert_class": "success",
        }
    if probability >= 0.5:
        return {
            "classification": "Intermediate / review",
            "status": "intermediate",
            "alert_class": "warning",
        }
    return {
        "classification": "Low antifungal signal",
        "status": "low",
        "alert_class": "danger",
    }


def sanitize_threshold(value):
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    return min(0.99, max(0.01, threshold))
