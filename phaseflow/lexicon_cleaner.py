"""
PhaseFlow v1.3 — lexicon_cleaner.py
Leksikona Tīrītājs un Zelta Vārdu Integrētājs.

Plūsma:
  1. Ielādē esošo phaseflow_mega_lexicon.npz
  2. Izfiltrē "netīros" vārdus pēc morfoloģiskiem likumiem
  3. Pievieno klāt "Zelta vārdus" (bāzes formas: lietvārdi, darbības vārdi)
  4. Kodē jaunos vārdus ar Kuramoto oscilatoru (reusing K_matrix)
  5. Saglabā atjaunināto phaseflow_mega_lexicon.npz

Izmantošana:
    python phaseflow/lexicon_cleaner.py
    python phaseflow/lexicon_cleaner.py --input phaseflow_mega_lexicon.npz --dry-run
"""

from __future__ import annotations

import sys
import os
import re
import argparse

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from smart_mirror import SmartMirror
from mass_encoder import _compute_signature, OUTPUT_PATH

# ═══════════════════════════════════════════════════════════════
#  FILTRĒŠANAS LIKUMI
# ═══════════════════════════════════════════════════════════════

# Galotnes, kas norāda uz netīrām ģenerētām formām
# (no download_corpus inflektora un manuskripta apmācības artefaktiem)
_BAD_SUFFIXES = (
    "tī",                           # garš patskanis + ī (p.e. "rakstītī")
    "sū",                           # artefakts ar ū
    "dams", "dame", "dami", "dama", # divdabju galotnes
    "ojot", "ējot",                 # gerundijs
    "šanas", "šanā",                # verbālvārdu locījumi (ne bāze)
    "isk",                          # "elpoisk" tips artefakts
    "īus", "āus", "ūus",            # nereālas galotnes
    "tiesos", "iesos",              # nereālas atstarojošās formas
    # ── Manuskripta apmācības atliekas ───────────────────────
    "oto", "āsi", "ieim",           # manuskripta locīšanas artefakti
    "ošos", "ājos", "ējos",         # atstarojošie divdabji
    "āmies", "āties",               # atstarojošie darbības vārdi
)

# Regulārā izteiksme — atļautie Latvijas alfabēta simboli + defise
_LATVIAN_RE = re.compile(
    r"^[a-zA-ZāčēģīķļņšūžĀČĒĢĪĶĻŅŠŪŽ\-]+$"
)

# Dubultoti / nereāli patskaņu pāri latviešu fonotaktikā
_BAD_VOWEL_RE = re.compile(
    r"(aa|ee|oo|ii|uu"       # ASCII dubultpatskaņi
    r"|āa|aā|ēe|eē|īi|iī|ūu|uū"   # garo + īso pats. dublikāts
    r"|āā|ēē|īī|ūū"          # dubulti garo patskaņu
    r")",
    re.IGNORECASE
)


def is_clean_word(word: str, max_len: int = 12) -> bool:
    """
    Atgriež True, ja vārds uzskatāms par "tīru" bāzes formu.

    Likumi:
    - garums: 2 ≤ len(word) ≤ max_len
    - tikai latviešu burti (ar diakritikām) un defise
    - nav transliterācijas artefaktu (aa, ee, oo, ii, uu)
    - nepabeidzas ar ģenerētajām galotnes shēmām
    """
    if not (2 <= len(word) <= max_len):
        return False
    if not _LATVIAN_RE.match(word):
        return False
    if _BAD_VOWEL_RE.search(word):
        return False
    w_lower = word.lower()
    if any(w_lower.endswith(sfx) for sfx in _BAD_SUFFIXES):
        return False
    return True


# ═══════════════════════════════════════════════════════════════
#  ZELTA VĀRDU SARAKSTS — Latviešu bāzes formas
# ═══════════════════════════════════════════════════════════════

ZELTA_VARДИ: list[str] = [
    # ── Daba un kosmoss ───────────────────────────────────────
    "saule", "mēness", "zvaigzne", "debesis", "zeme", "uguns",
    "ūdens", "vējš", "gaiss", "koks", "akmens", "ezers",
    "upe", "jūra", "kalns", "mežs", "lauks", "pļava",
    "mākoņi", "lietus", "sniegs", "ledus", "vētra", "migla",
    "zāle", "puķe", "zieds", "sakne", "lapa", "auglis",
    "putns", "zivs", "zvērs", "sēne", "lapiņa",

    # ── Gaisma un tumsa ──────────────────────────────────────
    "gaisma", "tumsa", "ēna", "stars", "spīdums", "mirdzums",
    "krāsa", "sarkans", "dzeltens", "zaļš", "zils", "balts",
    "melns", "sudrabs", "zelts", "dimants", "kristāls",

    # ── Cilvēks un dvēsele ───────────────────────────────────
    "dvēsele", "gars", "sirds", "prāts", "ķermenis", "elpa",
    "pulss", "asinis", "rokas", "acis", "seja", "balss",
    "smaids", "asara", "sāpes", "prieks", "miers", "bailes",
    "drosme", "mīlestība", "naids", "skumjas", "cerība",
    "sapnis", "atmiņa", "doma", "jūtas", "gribasspēks",

    # ── Attiecības ───────────────────────────────────────────
    "cilvēks", "bērns", "māte", "tēvs", "brālis", "māsa",
    "draugs", "ienaidnieks", "mīļotais", "skolotājs", "meistars",
    "ceļabiedrs", "svešinieks", "kopiena", "tauta",

    # ── Darbības vārdi — bāzes forma ─────────────────────────
    "iet", "būt", "skriet", "lidot", "krist", "celties",
    "runāt", "klusēt", "smaidīt", "raudāt", "smieties",
    "dziedāt", "dejot", "rakstīt", "lasīt", "domāt",
    "just", "redzēt", "dzirdēt", "elpot", "dzīvot", "mirt",
    "dzimt", "augt", "plaukt", "ziedēt", "radīt", "veidot",
    "apvienot", "dalīt", "iznīcināt", "pārveidot", "mācīt",
    "meklēt", "atrast", "pazaudēt", "dot", "ņemt", "turēt",
    "atbrīvot", "saistīt", "atvērt", "aizvērt", "nākt", "iet",

    # ── Laiks ────────────────────────────────────────────────
    "rīts", "vakars", "diena", "nakts", "tagad", "pagātne",
    "nākotne", "sekunde", "minūte", "stunda", "gads", "mūžs",
    "sākums", "beigas", "ritms", "plūsma", "kustība",

    # ── Telpa ────────────────────────────────────────────────
    "telpa", "vieta", "ceļš", "vārti", "robeža", "centrs",
    "māja", "templis", "pilsēta", "ciems", "tilts", "vērtne",

    # ── Filozofija un garīgums ────────────────────────────────
    "patiesība", "ilūzija", "vienotība", "gudrība", "zināšana",
    "apziņa", "cēlonis", "sekas", "misija", "brīvība",
    "atmodа", "apgaismība", "transmutācija", "evolūcija",
    "harmonija", "rezonanse", "frekvence", "vilnis", "fāze",
    "sakārtotība", "haoss", "kārtība", "likums", "daba",

    # ── Ontoloģiskie arhetīpi ─────────────────────────────────
    "viens", "divi", "trīs", "četri", "pieci",
    "seši", "septiņi", "astoņi", "deviņi", "desmit",
    "vienpadsmit", "divpadsmit",
    "dualitāte", "trisvienība", "matrica", "spirāle",
    "arhetips", "simbols", "zīme", "kods", "mīts",

    # ── Sakrālie skaļumi ─────────────────────────────────────
    "AUM", "OM", "MA", "SAT", "NAM", "TAO",
    "mantra", "meditācija", "lūgšana", "rituāls",

    # ── Joks un spēle ────────────────────────────────────────
    "joks", "smiekli", "spēle", "izrāde", "mīkla", "paradokss",
    "ironija", "humors", "absurds", "brīnums", "pārsteigums",

    # ── Kvalitatīvie ─────────────────────────────────────────
    "liels", "mazs", "ātrs", "lēns", "silts", "auksts",
    "gaišs", "tumšs", "smags", "viegls", "stingrs", "mīksts",
    "jauns", "vecs", "dzīvs", "brīvs", "vientuļš", "kopīgs",
    "dziļš", "sekls", "plaši", "šauri", "tāls", "tuvs",
]


# ═══════════════════════════════════════════════════════════════
#  GALVENĀ TĪRĪŠANAS FUNKCIJA
# ═══════════════════════════════════════════════════════════════

def clean_and_rebuild(
    input_path: str = OUTPUT_PATH,
    output_path: str | None = None,
    max_len: int = 12,
    dry_run: bool = False,
) -> tuple[list[str], np.ndarray]:
    """
    Ielādē, filtrē un paplašina leksikonu ar Zelta vārdiem.

    Parametri
    ---------
    input_path  : .npz fails ar esošo leksikonu
    output_path : kur saglabāt (None → pārraksta input_path)
    max_len     : maksimālais vārda garums (noklusēts 12)
    dry_run     : True → tikai parāda statistiku, nesaglabā

    Atgriež
    -------
    (words, phases_matrix) — attīrītais + paplašinātais leksikons
    """
    if output_path is None:
        output_path = input_path

    # ── 1. Ielādē esošo leksikonu ────────────────────────────
    print(f"\n[Cleaner] Ielādē: {input_path}")
    data = np.load(input_path, allow_pickle=True)
    orig_words: list[str] = list(data["words"])
    orig_phases: np.ndarray = data["phases_matrix"].astype(np.float64)
    n_osc = orig_phases.shape[1]
    print(f"          Esošie vārdi: {len(orig_words)}")

    # ── 2. Filtrē "netīros" vārdus ───────────────────────────
    kept_words: list[str] = []
    kept_phases: list[np.ndarray] = []
    removed: list[str] = []

    for w, ph in zip(orig_words, orig_phases):
        if is_clean_word(w, max_len=max_len):
            kept_words.append(w)
            kept_phases.append(ph)
        else:
            removed.append(w)

    print(f"[Cleaner] Saglabāti : {len(kept_words)}  (noņemti: {len(removed)})")
    if removed[:10]:
        print(f"          Piemēri noņemtiem: {removed[:10]}")

    # ── 3. Apvieno ar Zelta vārdiem (izvairoties no dublikātiem) ─
    existing_set: set[str] = set(kept_words)
    new_words_to_encode: list[str] = [
        w.lower() for w in ZELTA_VARДИ
        if w.lower() not in existing_set and w not in existing_set
        and is_clean_word(w.lower(), max_len=max_len + 4)  # Zelta vārdiem mīkstāks garums
    ]
    print(f"[Cleaner] Jauni Zelta vārdi kodējami: {len(new_words_to_encode)}")

    if new_words_to_encode and not dry_run:
        # ── 4. Ielādē SmartMirror ar K_matrix ──────────────
        print("[Cleaner] Inicializē SmartMirror ar K_matrix atmiņu ...")
        mirror = SmartMirror(
            verbose=False,
            enable_learning=False,
            use_lexicon=False,
            memory_path="phaseflow_mirror_memory.npz",
        )

        k_matrix = None
        if mirror._persistent_osc is not None:
            k_matrix = mirror._persistent_osc.K_matrix.copy()
            mean_k = float(np.mean(k_matrix[~np.eye(len(k_matrix), dtype=bool)]))
            print(f"          K_matrix: K̄={mean_k:.4f}")

        # ── 5. Kodē Zelta vārdus ─────────────────────────────
        new_phases: list[np.ndarray] = []
        for i, w in enumerate(new_words_to_encode):
            ph = _compute_signature(
                w.lower(), mirror.encoder, mirror.K, k_matrix, n=n_osc
            )
            new_phases.append(ph)
            if (i + 1) % 50 == 0 or (i + 1) == len(new_words_to_encode):
                print(f"          Kodēts {i + 1}/{len(new_words_to_encode)} ...", end="\r")
        print()

        # ── 6. Apvieno: Zelta vārdi PIRMIE, tad filtrētie esošie ─
        final_words = (
            [w.lower() for w in new_words_to_encode] + kept_words
        )
        final_phases = np.vstack(
            [np.array(new_phases)] + [np.array(kept_phases)]
        )
    else:
        final_words = kept_words
        final_phases = np.array(kept_phases) if kept_phases else orig_phases[:0]

    # Noņem dublikātus (saglabā pirmo ierakstu)
    seen: set[str] = set()
    dedup_words: list[str] = []
    dedup_idx: list[int] = []
    for i, w in enumerate(final_words):
        if w not in seen:
            seen.add(w)
            dedup_words.append(w)
            dedup_idx.append(i)
    final_phases = final_phases[dedup_idx]
    final_words = dedup_words

    print(f"[Cleaner] Galīgais leksikons: {len(final_words)} vārdi")

    if not dry_run:
        # ── 7. Saglabā .npz ──────────────────────────────────
        np.savez_compressed(
            output_path,
            words=np.array(final_words, dtype=object),
            phases_matrix=final_phases.astype(np.float32),
        )
        print(f"[Cleaner] Saglabāts: {output_path}")
    else:
        print("[Cleaner] Dry-run — fails nav mainīts.")

    return final_words, final_phases


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PhaseFlow Leksikona Tīrītājs — filtrē un pievieno Zelta vārdus"
    )
    parser.add_argument(
        "--input", default=OUTPUT_PATH,
        help=f"Ievades .npz fails (noklusēts: {OUTPUT_PATH})"
    )
    parser.add_argument(
        "--output", default=None,
        help="Izvades .npz fails (noklusēts: pārraksta --input)"
    )
    parser.add_argument(
        "--max-len", type=int, default=12,
        help="Maksimālais vārda garums (noklusēts: 12)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Tikai parāda statistiku, nesaglabā"
    )
    args = parser.parse_args()

    words, phases = clean_and_rebuild(
        input_path=args.input,
        output_path=args.output,
        max_len=args.max_len,
        dry_run=args.dry_run,
    )

    print(f"\n  Gatavs! {len(words)} tīri vārdi leksikonā.")
    print(f"  Pirmie 10: {words[:10]}")
