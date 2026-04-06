"""
PhaseFlow v1.2 — mass_encoder.py
Masveida Vārdu Kodētājs: "sagremo" lielus vārdu sarakstus
un eksportē θ_final fāžu matricu .npz failā.

Izmantošana:
    python phaseflow/mass_encoder.py
    → izveido phaseflow_mega_lexicon.npz

Vai kā modulis:
    from phaseflow.mass_encoder import encode_all_words, LATVIAN_WORDS
    words, matrix = encode_all_words(mirror, LATVIAN_WORDS)
"""

from __future__ import annotations

import sys
import os
import re
import time
import unicodedata

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from core import KuramotoOscillator, BASE_HARMONICS
from smart_mirror import SmartMirror, TextPhaseEncoder

OUTPUT_PATH = "phaseflow_mega_lexicon.npz"

# Rezerves saraksts — aktīvs tikai bez --words faila
# (pārdefinē zemāk pēc LATVIAN_WORDS definīcijas, ja download_corpus nav)
_FALLBACK_WORDS: list[str] = []  # tiek aizpildīts zemāk
try:
    from download_corpus import BASE_WORDS as _FALLBACK_WORDS
except ImportError:
    pass  # _FALLBACK_WORDS tiks norādīts uz LATVIAN_WORDS pēc tā definīcijas

# ═══════════════════════════════════════════════════════════════
#  IEKŠĒJAIS REZERVES SARAKSTS (~200 vārdi)
# ═══════════════════════════════════════════════════════════════

LATVIAN_WORDS: list[str] = [
    # ── Darbības vārdi ────────────────────────────────────────
    "iet", "but", "skriet", "lidot", "krist", "celties", "doties",
    "runaat", "kluseet", "smaidiit", "raudat", "smieties", "dziedaat",
    "deejot", "rakstiet", "lasiet", "domaat", "juust", "redzeeet",
    "dzirdeet", "pieskarties", "elpot", "dziivot", "mirt", "dzimt",
    "aug", "plaukt", "ziedeet", "augt", "sarukt", "izplatiitiies",
    "apvienoties", "daliitiies", "radiet", "izniciinaat", "paarveideeet",

    # ── Lietvārdi — Daba un kosmoss ───────────────────────────
    "koks", "akmens", "debess", "ezers", "upite", "ugunskurs",
    "elpa", "laiks", "telpa", "rits", "vakars", "diena", "nakts",
    "kustiba", "klusums", "saule", "menness", "zvaigzne", "zeme",
    "uguns", "udens", "vejsh", "kalns", "jura", "mezhs", "sekla",
    "puke", "zieds", "sakne", "lapa", "miza", "auglis", "sieksta",
    "putns", "zivs", "zvers", "kukaiinis", "sene", "saamls",
    "migla", "lietus", "sniegs", "ledus", "veetra", "vaaravikne",

    # ── Lietvārdi — Ķermenis un sajūtas ─────────────────────
    "sirds", "prrats", "dvesele", "acis", "rokas", "kajas",
    "elposhana", "pulss", "asinis", "kauls", "aaada", "nervi",
    "smadzenes", "skaņa", "gaisma", "tumsiba", "siltums", "aukstums",

    # ── Lietvārdi — Laiks ────────────────────────────────────
    "sekunde", "minute", "stunda", "diennakts", "nedelja",
    "mendnesis", "gads", "gadsimts", "tukstosgads", "mūžs",
    "sakums", "beigas", "tagad", "pagaatne", "naakotne",
    "ritms", "mierstiiba", "atpuuta", "plūsma",

    # ── Lietvārdi — Cilvēks un sabiedrība ───────────────────
    "cilveks", "beerniins", "sieva", "virs", "giimene",
    "draugs", "skolotaajs", "raadittaajs", "vadonuis", "dziednieks",
    "meekslinieks", "reeknieks", "filosofs", "mistiikis", "ceeļinieks",
    "kopiena", "pilseta", "ciems", "maajas", "templis",

    # ── Filozofija un garīgums ────────────────────────────────
    "patiesiiba", "iluzija", "vieniiба", "pluraliiба",
    "ceļsh", "varti", "robezhа", "centrs", "sakums",
    "atmodа", "apgaismiiба", "transmutacija", "evolucija",
    "atbriivoshana", "savienojums", "miilestiiба", "gudriiба",
    "harmonija", "rezonaana", "vibracija", "faze",

    # ── Ontoloģiskie arhetīpi ─────────────────────────────────
    "viens", "divi", "tris", "cetri", "pieci",
    "sesi", "septini", "astoni", "devini", "desmit",
    "vienpadsmit", "divpadsmit", "trinpadsmit",
    "singularitate", "dualitate", "trisvieniba", "struktura",
    "mistersija", "bezgaligba", "pabeigtiba", "matrica", "alkimija",
    "feniks", "oktava", "chakra", "spirale", "eteris", "zodiaks",

    # ── Kvalitātes ────────────────────────────────────────────
    "liels", "mazs", "aaatrs", "leens", "silts", "auksts",
    "gaaishs", "tumsh", "smags", "viegls", "stingrs", "mīksts",
    "jauns", "vecs", "dzīvs", "mirts", "brīvs", "saiistīts",
    "vientuligsh", "kopienigsh", "harmonigsh", "chaotisks",

    # ── Skaita vārdi un apstākļa vārdi ───────────────────────
    "tagadne", "šeit", "tur", "kopaa", "viens", "daudz", "maz",
    "aizvien", "joprojaam", "jau", "veeēl", "šodien", "vakar",

    # ── Mantras un sakrālie elementi ─────────────────────────
    "AUM", "OM", "MA", "RAM", "YAM", "HAM", "LAM", "VAM",
    "SAT", "NAM", "TAO", "ZEN",
    "zelta", "sudrabs", "dimants", "kristals", "smaragds",
    "rezonanse", "frekvence", "amplituuda", "vilnis", "lauks",
    "kvants", "fotons", "elektrons", "atoms", "kods",
]

# Ja download_corpus nebija pieejams, izmanto šo sarakstu kā rezervi
if not _FALLBACK_WORDS:
    _FALLBACK_WORDS = LATVIAN_WORDS

# ═══════════════════════════════════════════════════════════════
#  VĀRDU IELĀDE NO FAILA
# ═══════════════════════════════════════════════════════════════

def load_words_from_file(filepath: str, limit: int | None = None) -> list[str]:
    """
    Nolasa un attīra vārdus no teksta faila.
    Atbalsta "vārds" un "vārds frekvence" formātus (viens pa rindiņai).
    """
    words: list[str] = []
    seen: set[str] = set()
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            token = line.strip().split()[0] if line.strip() else ""
            # Minimāla attīrīšana: tikai burti un defise
            token = re.sub(r"[^a-zA-ZāčēģīķļņšūžĀČĒĢĪĶĻŅŠŪŽ\-]", "", token)
            token = unicodedata.normalize("NFC", token.strip("-").lower())
            if 2 <= len(token) <= 30 and token not in seen:
                seen.add(token)
                words.append(token)
            if limit and len(words) >= limit:
                break
    return words


# ═══════════════════════════════════════════════════════════════
#  KODĒŠANAS KODOLS
# ═══════════════════════════════════════════════════════════════

def _compute_signature(
    word: str,
    encoder: TextPhaseEncoder,
    K: float,
    k_matrix: np.ndarray | None,
    n: int = 5,
) -> np.ndarray:
    """
    Palaiž Kuramoto integrāciju vienam vārdam.

    Returns
    -------
    theta_final : (n,) — stabilie oscilatoru leņķi
    """
    init_theta = encoder.text_to_init_phases(word, n_oscillators=n)

    osc = KuramotoOscillator(
        n_oscillators=n,
        K=K,
        theta0=init_theta,
        seed=None,
        use_matrix=(k_matrix is not None),
    )
    if k_matrix is not None:
        osc.K_matrix = k_matrix.copy()

    # Integrē līdz stabilitātei (max 500 soļi, dt=0.02)
    prev_R = 0.0
    for _ in range(500):
        osc.step(0.02)
        R, _ = osc.order_parameter()
        if abs(R - prev_R) < 1e-4:
            break
        prev_R = R

    return osc.theta.copy()


def encode_all_words(
    mirror: SmartMirror,
    words: list[str],
    output_path: str = OUTPUT_PATH,
    batch_size: int = 50,
) -> tuple[list[str], np.ndarray]:
    """
    Kodē visu vārdu sarakstu uz θ_final fāžu matric un saglabā .npz failā.

    Parametri
    ---------
    mirror      : SmartMirror ar ielādētu K_matrix atmiņu
    words       : vārdu saraksts
    output_path : mērķa .npz faila ceļš
    batch_size  : partijas lielums (progress indikatora grupēšanai)

    Atgriež
    -------
    (words, phases_matrix)
        phases_matrix : (N_vārdi, 5) masīvs ar θ_final katram vārdam
    """
    n_osc = len(BASE_HARMONICS)  # 5
    n_words = len(words)

    # Iegūst K_matrix no persistentā oscilatora
    k_matrix = None
    if hasattr(mirror, "_persistent_osc") and mirror._persistent_osc is not None:
        k_matrix = mirror._persistent_osc.K_matrix.copy()
        mean_k = float(np.mean(k_matrix[~np.eye(len(k_matrix), dtype=bool)]))
        print(f"  K_matrix ielādēta: K̄={mean_k:.6f}")
    else:
        print("  [!] Nav K_matrix — izmanto vienmērīgu sakabi K={mirror.K}")

    encoder = mirror.encoder
    K = mirror.K

    # tqdm — ja pieejams, citādi rezerves josla
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    print(f"\n{'═'*60}")
    print(f"  MASVEIDA KODĒŠANA — {n_words} vārdi → {n_osc}D fāžu paraksti")
    print(f"  Izeja: {output_path}")
    if use_tqdm:
        print(f"  Progresa josla: tqdm")
    print(f"{'═'*60}\n")

    phases_list: list[np.ndarray] = []
    t_start = time.time()

    if use_tqdm:
        iterator = tqdm(
            words,
            desc="  Kodē",
            unit=" vārds",
            ncols=70,
            bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
    else:
        iterator = words

    for i, word in enumerate(iterator if use_tqdm else words, 1):
        sig = _compute_signature(word, encoder, K, k_matrix, n=n_osc)
        phases_list.append(sig)

        if not use_tqdm:
            # Rezerves progresa indikators
            if i % batch_size == 0 or i == n_words:
                elapsed = time.time() - t_start
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (n_words - i) / rate if rate > 0 else 0.0
                pct = 100.0 * i / n_words
                bar_filled = int(pct / 5)
                bar = "█" * bar_filled + "░" * (20 - bar_filled)
                print(
                    f"  [{bar}] {pct:5.1f}%  "
                    f"{i:>6}/{n_words}  "
                    f"{rate:5.1f} vārdi/s  "
                    f"ETA: {eta:4.0f}s"
                )

    phases_matrix = np.array(phases_list, dtype=np.float64)  # (N, 5)

    # Saglabā .npz
    np.savez_compressed(
        output_path,
        words=np.array(words, dtype=object),
        phases_matrix=phases_matrix,
    )

    elapsed_total = time.time() - t_start
    print(f"\n{'─'*60}")
    print(f"  ✓ Saglabāts: {output_path}")
    print(f"    Vārdu skaits : {n_words}")
    print(f"    Matrica forma: {phases_matrix.shape}  (vārdi × oscilatoru fāzes)")
    print(f"    Kopējais laiks: {elapsed_total:.1f}s")
    print(f"{'─'*60}\n")

    return words, phases_matrix


# ═══════════════════════════════════════════════════════════════
#  IEEJAS PUNKTS
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PhaseFlow — Masveida Vārdu Kodētājs")
    parser.add_argument(
        "--words",
        default=None,
        help="Ceļš uz latvian_words.txt (noklusējums: izmanto iekšējo sarakstu)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maksimālais vārdu skaits no faila"
    )
    parser.add_argument(
        "--output", default=OUTPUT_PATH,
        help=f"Izejas .npz fails (noklusējums: {OUTPUT_PATH})"
    )
    args = parser.parse_args()

    print("PhaseFlow — Masveida Vārdu Kodētājs")
    print("Ielādē SmartMirror ar K_matrix atmiņu ...\n")

    mirror = SmartMirror(
        verbose=False,
        enable_learning=False,
        use_lexicon=False,
    )

    # Ielādē vārdus
    if args.words and os.path.exists(args.words):
        print(f"Nolasa vārdus no: {args.words}")
        word_list = load_words_from_file(args.words, limit=args.limit)
        print(f"Ielādēti {len(word_list)} vārdi no faila.\n")
    else:
        if args.words:
            print(f"[!] Fails nav atrasts: {args.words}")
            print(f"    Izmanto iekšējo bāzes sarakstu ({len(_FALLBACK_WORDS)} vārdi).\n")
        else:
            print(f"Izmanto iekšējo bāzes sarakstu ({len(_FALLBACK_WORDS)} vārdi).")
            print(f"Tip: python phaseflow/download_corpus.py  ← ģenerē latvian_words.txt\n")
        word_list = [w.lower() for w in _FALLBACK_WORDS]

    if args.limit and len(word_list) > args.limit:
        word_list = word_list[:args.limit]

    words, matrix = encode_all_words(
        mirror=mirror,
        words=word_list,
        output_path=args.output,
    )

    print(f"Gatavs! Kopā {len(words)} vārdi iekodēti.")
    print(f"Nākamais solis: python phaseflow/fast_lexicon.py")
