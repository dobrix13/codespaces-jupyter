"""
PhaseFlow v1.2 — download_corpus.py
Lielā Latviešu Korpusa Iegūšana (~10 000–50 000 vārdi).

Stratēģija (3 pakāpes, izmēģina secīgi):
  1. HTTP lejupielāde no publiski pieejamiem vārdu biežuma sarakstiem
     (Leipzig Corpora Collection, OPUS, vai GitHub latviešu NLP resursi).
  2. NLTK/vārdu ģeneratora rezerves ceļš, ja HTTP neizdodas.
  3. Locīšanas ģenerators — paplašina bāzes vārdu sarakstu ar biežākajām
     latviešu locījuma galotnēm (~8x reizinātājs bez ārējiem avotiem).

Izeja:
  latvian_words.txt  — viens vārds rindā, tīrs, bez dublikātiem,
                        sakārtots pēc biežuma (ja pieejams) vai alfabētiski.

Izmantošana:
    python phaseflow/download_corpus.py
    python phaseflow/download_corpus.py --limit 10000
    python phaseflow/download_corpus.py --source generate
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
import urllib.request
import urllib.error

# ═══════════════════════════════════════════════════════════════
#  PUBLISKO AVOTU SARAKSTS
#  Tiek izmēģināti secīgi — pirmais veiksmīgais tiek izmantots.
# ═══════════════════════════════════════════════════════════════

REMOTE_SOURCES: list[dict] = [
    # Leipzig Corpora Collection — latviešu valoda, 10k biežākie vārdi
    {
        "name": "Leipzig LV 10k",
        "url": (
            "https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
            "master/content/2018/lv/lv_50k.txt"
        ),
        "format": "freq",   # "word freq" pa rindiņām
    },
    # OpenSubtitles latviešu biežuma saraksts (Hermit Dave)
    {
        "name": "OpenSubtitles LV 50k",
        "url": (
            "https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
            "master/content/2016/lv/lv_full.txt"
        ),
        "format": "freq",
    },
    # Rezerve — vienkāršs alfabētisks saraksts no OPUS latviešu korpusa
    {
        "name": "OPUS LV wordlist",
        "url": (
            "https://raw.githubusercontent.com/stopwords-iso/stopwords-lv/"
            "master/stopwords-lv.json"
        ),
        "format": "json_list",
    },
]

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "latvian_words.txt",
)

# ═══════════════════════════════════════════════════════════════
#  BĀZES LATVIEŠU VĀRDU KRĀJUMS (rezerves ģeneratoram)
# ═══════════════════════════════════════════════════════════════

BASE_WORDS: list[str] = [
    # Darbības vārdi
    "iet", "nākt", "būt", "kļūt", "skriet", "lidot", "krist", "celties",
    "doties", "runāt", "klusēt", "smaidīt", "raudāt", "smieties", "dziedāt",
    "dejot", "rakstīt", "lasīt", "domāt", "just", "redzēt", "dzirdēt",
    "pieskarties", "elpot", "dzīvot", "mirt", "dzimt", "augt", "plaukt",
    "ziedēt", "sarukt", "izplatīties", "apvienoties", "dalīties", "radīt",
    "iznīcināt", "pārveidot", "mācīties", "zināt", "saprast", "mīlēt",
    "ienīst", "atcerēties", "aizmirst", "snaust", "gulēt", "mosties",
    "ēst", "dzert", "staigāt", "braukt", "skriet", "nēsāt", "dot", "ņemt",
    "atvērt", "aizvērt", "sākt", "beigt", "turpināt", "apstāties",
    "palīdzēt", "traucēt", "gaidīt", "meklēt", "atrast", "pazaudēt",
    "cīnīties", "uzvarēt", "zaudēt", "kalpot", "valdīt", "dziedināt",
    "sāpēt", "priecāties", "bēdāties", "baidīties", "cerēt", "ticēt",

    # Lietvārdi — daba
    "saule", "mēness", "zvaigzne", "zeme", "debesis", "uguns", "ūdens",
    "gaiss", "vējš", "kalns", "jūra", "upe", "ezers", "mežs", "lauks",
    "zāle", "koks", "puķe", "zieds", "sakne", "lapa", "miza", "auglis",
    "sēkla", "putns", "zivs", "zvērs", "kukainis", "sēne", "mākoņi",
    "migla", "lietus", "sniegs", "ledus", "vētra", "varavīksne", "akmeņi",
    "smiltis", "māls", "zelts", "sudrabs", "dimants", "kristāls",

    # Lietvārdi — ķermenis
    "sirds", "prāts", "dvēsele", "acis", "rokas", "kājas", "galva",
    "mutes", "ausīs", "deguns", "kakls", "mugura", "vēders", "pleci",
    "pirksti", "nagi", "mati", "āda", "kauls", "asinis", "elpa", "pulss",
    "smadzenes", "nervi", "muskuļi",

    # Lietvārdi — laiks
    "sekunde", "minūte", "stunda", "diena", "nakts", "rīts", "vakars",
    "nedēļa", "mēnesis", "gads", "gadsimts", "mūžs", "brīdis", "laiks",
    "pagātne", "tagadne", "nākotne", "sakums", "beigas", "ritms",

    # Lietvārdi — cilvēks un sabiedrība
    "cilvēks", "bērns", "sieva", "vīrs", "ģimene", "draugs", "ienaidnieks",
    "skolotājs", "skolēns", "ārsts", "mākslinieks", "rakstnieks", "dzejnieks",
    "vadonis", "karalis", "karaliene", "kalps", "sargs", "tirgotājs",
    "zemnieks", "pilsēta", "ciems", "mājas", "templis", "skola", "tirgus",
    "ceļš", "tilts", "robeža", "valsts", "tauta", "valoda", "kultūra",

    # Filozofija un garīgums
    "patiesība", "ilūzija", "vienotība", "daudzveidība", "harmonija",
    "rezonanse", "fāze", "vibrācija", "enerģija", "frekvence", "vilnis",
    "gaisma", "tumsa", "mīlestība", "bailes", "prieks", "sāpes", "cerība",
    "ticība", "gudrība", "miers", "brīvība", "atbildība", "sirdsapziņa",
    "apziņa", "pašapziņa", "intuīcija", "sapnis", "realitāte", "brīnums",
    "noslēpums", "patiesums", "skaistums", "labums", "taisnīgums",

    # Skaitļi un kvantitāte
    "viens", "divi", "trīs", "četri", "pieci", "seši", "septiņi",
    "astoņi", "deviņi", "desmit", "simts", "tūkstotis", "miljons",
    "pirmais", "otrais", "trešais", "pēdējais", "daudz", "maz", "visi",

    # Vietniekvārdi un apstākļa vārdi
    "es", "tu", "viņš", "viņa", "mēs", "jūs", "viņi", "kas", "ko",
    "kur", "kad", "kā", "kāpēc", "šeit", "tur", "tagad", "tad", "jau",
    "vēl", "arī", "tikai", "varbūt", "noteikti", "vienmēr", "nekad",

    # Īpašības vārdi
    "liels", "mazs", "ātrs", "lēns", "silts", "auksts", "gaišs", "tumšs",
    "smags", "viegls", "stingrs", "mīksts", "jauns", "vecs", "dzīvs",
    "brīvs", "vientuļš", "laimīgs", "bēdīgs", "stiprs", "vājš",
    "skaists", "neglīts", "gudrs", "muļķīgs", "drosmīgs", "bailīgs",
    "labs", "ļauns", "pareizs", "nepareizs", "īsts", "viltus",

    # Ontoloģiskie arhetīpi
    "singularitāte", "dualitāte", "trīsvienība", "struktūra",
    "mistērija", "bezgalība", "pabeigšana", "matrica", "alķīmija",
    "fēnikss", "oktāva", "čakra", "spirāle", "ēters", "zodiaks",
    "AUM", "OM", "MA", "TAO", "ZEN",
]

# Latviešu locīšanas galotnes (visbiežākās)
_LV_SUFFIXES: list[str] = [
    "s", "is", "us", "as", "a", "e", "i",          # nominatīvs
    "a", "es", "us", "as",                           # ģenitīvs
    "am", "ai", "um", "im",                          # datīvs
    "u", "i",                                        # akuzatīvs
    "ā", "ē", "ī", "ū",                              # lokatīvs
    "os", "ās", "ies", "us",                         # daudzskaitlis
    "ojot", "ējot", "ot", "dams", "dama",            # divdabji
    "šana", "šanas", "šanā",                         # verbālvārdi
    "isk", "nieks", "niece", "ums", "ums",           # atvasinājumi
]


# ═══════════════════════════════════════════════════════════════
#  TEKSTA ATTĪRĪŠANA
# ═══════════════════════════════════════════════════════════════

def clean_word(word: str) -> str | None:
    """
    Attīra vārdu:
    - noņem simbolus (pieturzīmes, ciparus, speciālos simbolus)
    - pārvērš uz mazajiem burtiem
    - noraida vārdus īsākus par 2 vai garākus par 30 simboliem
    - normalizē Unicode (NFD → NFC)

    Atgriež tīru vārdu vai None, ja tiek noraidīts.
    """
    word = word.strip()
    # Noņem visu, kas nav burti vai defise starp burtiem
    word = re.sub(r"[^a-zA-ZāčēģīķļņšūžĀČĒĢĪĶĻŅŠŪŽ\-]", "", word)
    word = re.sub(r"^-+|-+$", "", word)   # defise sākumā/beigās
    word = unicodedata.normalize("NFC", word)
    word = word.lower()
    if len(word) < 2 or len(word) > 30:
        return None
    return word


def load_words_from_file(filepath: str, limit: int | None = None) -> list[str]:
    """
    Nolasa un attīra vārdus no teksta faila (viens vārds rindā vai
    "vārds frekvence" formāts).
    """
    words: list[str] = []
    seen: set[str] = set()

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Atbalsta "vārds freq" un "vārds" formātus
            token = line.split()[0] if line.split() else ""
            cleaned = clean_word(token)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                words.append(cleaned)
            if limit and len(words) >= limit:
                break

    return words


# ═══════════════════════════════════════════════════════════════
#  LEJUPIELĀDES FUNKCIJAS
# ═══════════════════════════════════════════════════════════════

def _try_download(source: dict, limit: int | None, timeout: int = 15) -> list[str] | None:
    """Mēģina lejupielādēt vārdus no viena avota. Atgriež None neveiksmē."""
    url = source["url"]
    fmt = source["format"]
    name = source["name"]

    print(f"  Mēģina lejupielādēt: {name}")
    print(f"  URL: {url[:80]}{'...' if len(url) > 80 else ''}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PhaseFlow/1.2"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"  [!] Neizdevās: {e}")
        return None

    words: list[str] = []
    seen: set[str] = set()

    if fmt == "freq":
        # "vārds skaitlis" pa rindām
        for line in raw.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cleaned = clean_word(parts[0])
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                words.append(cleaned)
            if limit and len(words) >= limit:
                break

    elif fmt == "json_list":
        import json
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    cleaned = clean_word(str(item))
                    if cleaned and cleaned not in seen:
                        seen.add(cleaned)
                        words.append(cleaned)
                    if limit and len(words) >= limit:
                        break
        except json.JSONDecodeError:
            print(f"  [!] JSON parsēšanas kļūda.")
            return None

    if not words:
        print(f"  [!] Nav derīgu vārdu.")
        return None

    print(f"  ✓ Iegūti {len(words)} vārdi no '{name}'")
    return words


def download_words(limit: int | None = None) -> list[str]:
    """
    Mēģina lejupielādēt no visiem avotiem secīgi.
    Atgriež pirmo veiksmīgo rezultātu.
    """
    for source in REMOTE_SOURCES:
        words = _try_download(source, limit)
        if words:
            return words
    return []


# ═══════════════════════════════════════════════════════════════
#  LOCĪŠANAS ĢENERATORS (rezerves ceļš)
# ═══════════════════════════════════════════════════════════════

def generate_inflected_words(base_words: list[str], limit: int | None = None) -> list[str]:
    """
    Ģenerē vārdu formas, piemērojot biežākās latviešu galotnes.

    Stratēģija: noņem pēdējos 1-3 burtus no bāzes vārda,
    pievieno katra sufiksa variantu. Filtrē pēc minimālā garuma.

    Šis nav pilns locītājs — tas ģenerē aptuvenas formas, kas
    tomēr ir derīgas Kuramoto fāžu kodēšanai (fonoloģiskā variācija).
    """
    results: list[str] = []
    seen: set[str] = set(base_words)

    for word in base_words:
        if len(word) < 4:
            continue
        # Izvelk celmu (noņem pēdējos 0-3 burtus)
        for cut in range(0, min(4, len(word) - 2)):
            stem = word[: len(word) - cut] if cut > 0 else word
            for suf in _LV_SUFFIXES:
                candidate = stem + suf
                if candidate not in seen and 3 <= len(candidate) <= 20:
                    seen.add(candidate)
                    results.append(candidate)
                if limit and (len(results) + len(base_words)) >= limit:
                    return base_words + results

    return base_words + results


# ═══════════════════════════════════════════════════════════════
#  GALVENĀ FUNKCIJA
# ═══════════════════════════════════════════════════════════════

def build_corpus(
    output_path: str = OUTPUT_PATH,
    limit: int | None = 50_000,
    source: str = "auto",
) -> list[str]:
    """
    Izveido latviešu vārdu korpusu un saglabā output_path failā.

    Parametri
    ---------
    output_path : mērķa .txt faila ceļš
    limit       : maksimālais vārdu skaits (None = bez ierobežojuma)
    source      : "auto" | "download" | "generate"
                  "auto"     — vispirms lejupielāde, neveiksmē ģenerēšana
                  "download" — tikai lejupielāde
                  "generate" — tikai locīšanas ģenerators

    Atgriež
    -------
    list[str] — galīgais vārdu saraksts
    """
    print("=" * 62)
    print("  PhaseFlow — Latviešu Korpusa Ģenerators")
    print("=" * 62)
    print(f"  Avots  : {source}")
    print(f"  Limits : {limit if limit else 'bez ierobežojuma'}")
    print(f"  Izeja  : {output_path}\n")

    words: list[str] = []

    # ── 1. sadaļa: HTTP lejupielāde ─────────────────────
    if source in ("auto", "download"):
        print("  [1/3] Mēģina HTTP lejupielādi ...")
        words = download_words(limit=limit)

    # ── 2. sadaļa: rezerve — ģenerators ─────────────────
    if not words and source in ("auto", "generate"):
        print(f"\n  [2/3] HTTP neizdevās vai izlaists. Izmanto locīšanas ģeneratoru ...")
        print(f"        Bāzes vārdi: {len(BASE_WORDS)}")
        words = generate_inflected_words(BASE_WORDS, limit=limit)
        print(f"        Pēc locījumiem: {len(words)} vārdi")

    # ── 3. sadaļa: ja viss citur neizdevās — bāze vien ──
    if not words:
        print("\n  [3/3] Izmanto tikai bāzes vārdu sarakstu ...")
        words = [w.lower() for w in BASE_WORDS]

    # Pielietojam limitu
    if limit and len(words) > limit:
        words = words[:limit]

    # Nodrošina dublikātu neesamību (saglabā kārtību)
    seen: set[str] = set()
    unique_words: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)
    words = unique_words

    # ── Saglabā failā ────────────────────────────────────
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(words) + "\n")

    print(f"\n{'─'*62}")
    print(f"  ✓ Saglabāts : {output_path}")
    print(f"    Vārdi     : {len(words)}")
    print(f"    Paraugs   : {words[:8]}")
    print(f"{'─'*62}")

    return words


# ═══════════════════════════════════════════════════════════════
#  IEEJAS PUNKTS
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lejupielādē vai ģenerē latviešu vārdu korpusu"
    )
    parser.add_argument(
        "--limit", type=int, default=50_000,
        help="Maksimālais vārdu skaits (noklusējums: 50000)"
    )
    parser.add_argument(
        "--source", choices=["auto", "download", "generate"], default="auto",
        help="Avotu stratēģija (noklusējums: auto)"
    )
    parser.add_argument(
        "--output", default=OUTPUT_PATH,
        help=f"Izejas faila ceļš (noklusējums: {OUTPUT_PATH})"
    )
    args = parser.parse_args()

    words = build_corpus(
        output_path=args.output,
        limit=args.limit,
        source=args.source,
    )
    print(f"\n  Nākamais solis:")
    print(f"    python phaseflow/mass_encoder.py --words {args.output}")
