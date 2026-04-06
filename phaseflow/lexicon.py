"""
PhaseFlow v1.1 — lexicon.py
Hologrāfiskais Leksikons (Holographic Lexicon).

Katram vārdam aprēķina "Fāžu Parakstu" — 5 Kuramoto oscilatoru
beigu leņķus pēc sinhronizācijas. Dekodēšana: cirkulārā distance
starp target_theta un leksikona parakstiem → tuvākie vārdi.

Lietošana:
    from lexicon import PhaseLexicon
    lex = PhaseLexicon()
    lex.build(mirror)          # aprēķina un saglabā
    words = lex.decode_to_words(theta, top_k=3)
"""

import json
import os
import numpy as np

# ─────────────────────────────────────────────────────────────
#  Importē PhaseFlow kodolu (atbalsta gan moduli, gan tiešu izpildi)
# ─────────────────────────────────────────────────────────────
try:
    from phaseflow.core import KuramotoOscillator, BASE_HARMONICS
    from phaseflow.smart_mirror import TextPhaseEncoder
except ImportError:
    from core import KuramotoOscillator, BASE_HARMONICS
    from smart_mirror import TextPhaseEncoder

# ═══════════════════════════════════════════════════════════════
#  VĀRDU KRĀJUMS — ~80 fundamentālie latviešu jēdzieni
# ═══════════════════════════════════════════════════════════════

CORE_WORDS: list[str] = [
    # Kosmoss un Daba
    "Gaisma", "Tumsiba", "Saule", "Menness", "Zvaigzne",
    "Zeme", "Debesis", "Uguns", "Udens", "Vejsh",
    "Kalns", "Jura", "Mežs", "Sekla", "Puķe",

    # Laiks un Plūsma
    "Sakums", "Beigas", "Tagad", "Pagātne", "Nakotne",
    "Pluusma", "Ritms", "Kustiba", "Mierstiiba", "Atpuuta",

    # Emocijas un Garīgums
    "Miilestiiба", "Bailes", "Prieks", "Saapes", "Ceriba",
    "Harmonija", "Rezononanse", "Dvesele", "Apzina", "Intuicija",
    "Pateiciiba", "Piedoshana", "Uzticiiба", "Drosme", "Pacietiiба",

    # Dzīves Jēdzieni
    "Cilveks", "Berniнs", "Sieva", "Virs", "Gimene",
    "Draugs", "Skolotajs", "Raditajs", "Feniks", "Gudriба",

    # Filozofija un Patiesība
    "Patiesiiба", "Iluzija", "Vieniiба", "Pluraliiба", "Singularitate",
    "Ceļsh", "Vaarti", "Robezhа", "Centrs", "Pereferija",
    "Atmodа", "Apgaismiiба", "Transmutacija", "Evolucija", "Revolucija",

    # Darbības un Stāvokļi
    "Elposhana", "Deeja", "Dziidаshana", "Klusums", "Skaņа",
    "Ir", "Nav", "Es", "Tu", "Mes",

    # Mantras un Sakrālie vārdi
    "AUM", "OM", "MA", "Savienojums", "Atbriivoshana",
    "Zelta", "Proporcija", "Rezonanse", "Faze", "Vibracija",

    # Dualitātes Integrācija — Alķīmija (ēna → gaisma)
    "Bailes", "Tumsa", "Ego", "Sapes", "Iluzija",
    "Gaisma", "Miilestiba", "Avots", "Parveide", "Pienemshana",
    "Speks", "Cels", "Dzimst", "Maciiба", "Integracija",

    # Kosmiskā Ontoloģija — Skaitļu Arhetīpi (1–13)
    "Viens",          # 1 — Singularitāte, Avots
    "Singularitate",  # 1 — Punkts bez dimensijām
    "Dualitate",      # 2 — Pretstats, Spogulis
    "Trisvieniba",    # 3 — Radīšana, Sintēze
    "Struktura",      # 4 — Matērija, Likums
    "Cilveks",        # 5 — Brīvā griba, Zelta proporcija
    "Harmonija",      # 6 — Makrokosmoss, Mīlestība
    "Mistersija",     # 7 — Garīgais ceļš, Čakras
    "Bezgaligba",     # 8 — Karma, Oktāva, Pārpilnība
    "Pabeigtiba",     # 9 — Cikla beigas, Atdeve
    "Varti",          # 10/11 — Jauns cikls, Vārti
    "Matrica",        # 12 — Kosmiskā arhitektūra
    "Alkimija",       # 13 — Fēnikss, Izlaušanās
    "Feniks",         # 13 — Nāve un atdzimšana
    "Oktava",         # 8 — Rezonanse augstākā frekvencē
    "Chakra",         # 7 — Enerģijas centri
    "Spirale",        # DNS, augšana
    "Eteris",         # 5 — Kvintesence
    "Zodiaks",        # 12 — Laiktelpas matrica
]

# ═══════════════════════════════════════════════════════════════
#  CIRKULĀRĀ DISTANCE (fāžu telpa)
# ═══════════════════════════════════════════════════════════════

def _circular_distance(theta_a: np.ndarray, theta_b: np.ndarray) -> float:
    """
    Kvadrātiskā Distance vienības aplī starp diviem fāžu vektoriem.

    D = Σ |e^{i·a_k} - e^{i·b_k}|²  (k = 0..N-1)

    Diapazone [0, 4·N].  Mazāks = tuvāks.
    """
    diff = np.exp(1j * theta_a) - np.exp(1j * theta_b)
    return float(np.sum(np.abs(diff) ** 2))


# ═══════════════════════════════════════════════════════════════
#  LEKSIKONS
# ═══════════════════════════════════════════════════════════════

class PhaseLexicon:
    """
    Hologrāfiskais Leksikons — vārdu → fāžu parakstu vārdnīca.

    Attributes
    ----------
    signatures : dict[str, np.ndarray]
        Vārds → (N,) fāžu paraksts θ_final
    """

    DEFAULT_PATH = "phaseflow_lexicon.json"

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = path
        self.signatures: dict[str, np.ndarray] = {}

    # ── Leksikona būvniecība ─────────────────────────────────

    def build(self,
              mirror,
              words: list[str] | None = None,
              force_rebuild: bool = False) -> None:
        """
        Aprēķina fāžu parakstus visiem vārdiem.

        Ja leksikons jau eksistē diskā un force_rebuild=False,
        ielādē to bez pārrēķina.

        Parameters
        ----------
        mirror        : SmartMirror instance (vai jebkas ar .encoder un .K)
        words         : saraksts ar vārdiem (noklusējums = CORE_WORDS)
        force_rebuild : True → pārrēķina pat ja fails eksistē
        """
        if words is None:
            words = CORE_WORDS

        if not force_rebuild and os.path.exists(self.path):
            self._load()
            # Papildina jaunu vārdu parakstus, ja kādi trūkst
            missing = [w for w in words if w not in self.signatures]
            if not missing:
                print(f"[Lexicon] Ielādēts no {self.path} ({len(self.signatures)} vārdi)")
                return
            print(f"[Lexicon] Papildina {len(missing)} jaunus vārdus ...")
            words = missing

        print(f"[Lexicon] Aprēķina fāžu parakstus {len(words)} vārdiem ...")
        encoder = mirror.encoder

        # Iegūst K_matrix no persistentā oscilatora (ja pieejams)
        k_matrix = None
        if hasattr(mirror, "_persistent_osc") and mirror._persistent_osc is not None:
            k_matrix = mirror._persistent_osc.K_matrix.copy()

        for word in words:
            sig = self._compute_signature(word, encoder, mirror.K, k_matrix)
            self.signatures[word] = sig

        self._save()
        print(f"[Lexicon] Saglabāts: {self.path} ({len(self.signatures)} vārdi)")

    def _compute_signature(self,
                           word: str,
                           encoder: "TextPhaseEncoder",
                           K: float,
                           k_matrix: np.ndarray | None) -> np.ndarray:
        """Palaiž Kuramoto integrāciju vienam vārdam, atgriež θ_final."""
        n = len(BASE_HARMONICS)
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

    # ── Saglabāšana / Ielāde ────────────────────────────────

    def _save(self) -> None:
        data = {w: sig.tolist() for w, sig in self.signatures.items()}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.signatures = {w: np.array(v) for w, v in data.items()}

    # ── Dekodēšana ──────────────────────────────────────────

    def decode_to_words(self,
                        target_theta: np.ndarray,
                        top_k: int = 3) -> list[tuple[str, float]]:
        """
        Atrod top_k tuvākos vārdus dotajam fāžu stāvoklim.

        Parameters
        ----------
        target_theta : (N,) fāžu masīvs (sinhronizācijas beigu stāvoklis)
        top_k        : cik tuvāko vārdu atgriezt

        Returns
        -------
        list[(vārds, distance)]  — sakārtots no mazākā uz lielāko
        """
        if not self.signatures:
            return []

        distances = [
            (word, _circular_distance(target_theta, sig))
            for word, sig in self.signatures.items()
        ]
        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    def __len__(self) -> int:
        return len(self.signatures)

    def is_built(self) -> bool:
        return bool(self.signatures)


# ═══════════════════════════════════════════════════════════════
#  ĀTRĀ INICIALIZĀCIJA
# ═══════════════════════════════════════════════════════════════

def build_lexicon(mirror, path: str = PhaseLexicon.DEFAULT_PATH,
                 force_rebuild: bool = False) -> PhaseLexicon:
    """Ātrā funkcija: izveido un atgriež aizpildītu PhaseLexicon."""
    lex = PhaseLexicon(path=path)
    lex.build(mirror, force_rebuild=force_rebuild)
    return lex


# ═══════════════════════════════════════════════════════════════
#  DEMONSTRĀCIJA
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from smart_mirror import SmartMirror

    print("=" * 55)
    print("  PhaseLexicon — Hologrāfiskais Vārdu Leksikons")
    print("=" * 55)

    mirror = SmartMirror(verbose=False, enable_learning=False)
    lex = build_lexicon(mirror, force_rebuild=True)

    print(f"\nLeksikons satur {len(lex)} vārdus.\n")

    # Testa fāze: ievadām "Gaisma" un meklējam tuvākos
    test_words = ["Gaisma", "AUM", "Bailes", "Cilveks"]
    for tw in test_words:
        sig = lex.signatures.get(tw)
        if sig is None:
            continue
        # Pievieno nelielu troksni
        noisy = sig + np.random.normal(0, 0.3, sig.shape)
        results = lex.decode_to_words(noisy, top_k=3)
        top_str = "  |  ".join(f"{w} ({d:.3f})" for w, d in results)
        print(f"  θ('{tw}') + troksnis  →  {top_str}")
