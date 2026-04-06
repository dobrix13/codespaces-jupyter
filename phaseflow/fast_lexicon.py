"""
PhaseFlow v1.2 — fast_lexicon.py
Ātrais K-Dimensional Tree Meklētājs cirkulārajā fāžu telpā.

Problēma: fāzes θ ∈ [0, 2π) ir APĻVEIDA — KDTree Eiklīda distance
          nezina, ka θ=0 un θ=2π ir viens punkts.

Risinājums — 10D iegulšana:
    θ  →  (cos θ, sin θ)
    5D apļi → 10D Eiklīda telpa, kurā attālums = cirkulārā distance

    |e^{ia} - e^{ib}|² = (cos a - cos b)² + (sin a - sin b)² = 2 - 2cos(a-b)

Tādējādi cKDTree.query() atgriež precīzu fāžu attālumu.

Izmantošana:
    from phaseflow.fast_lexicon import FastPhaseLexicon
    lex = FastPhaseLexicon.load("phaseflow_mega_lexicon.npz")
    results = lex.find_closest(theta_final, top_k=5)

Vai tiešā izpilde (demo):
    python phaseflow/fast_lexicon.py
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np
from scipy.spatial import cKDTree

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)


# ═══════════════════════════════════════════════════════════════
#  PALĪGFUNKCIJAS — fāžu ↔ Eiklīda telpa
# ═══════════════════════════════════════════════════════════════

def phases_to_euclidean(theta: np.ndarray) -> np.ndarray:
    """
    Pārveido fāžu vektoru (N,) uz Eiklīda iegulšanu (2N,).

    Katrs θ_k  →  [cos θ_k, sin θ_k]
    Tātad 5D apļu telpa → 10D Eiklīda telpa.

    Parametri
    ---------
    theta : (..., N) — fāžu masīvs radiānos (var būt 1D vai 2D)

    Atgriež
    -------
    (..., 2N) — Eiklīda iegulšana
    """
    cos_part = np.cos(theta)
    sin_part = np.sin(theta)
    # Savieno pa pēdējo asi: [..., cos0, cos1, ..., sin0, sin1, ...]
    return np.concatenate([cos_part, sin_part], axis=-1)


# ═══════════════════════════════════════════════════════════════
#  GALVENĀ KLASE
# ═══════════════════════════════════════════════════════════════

class FastPhaseLexicon:
    """
    Ātrais Fāžu Leksikons balstīts uz scipy.spatial.cKDTree.

    Atribūti
    --------
    words         : list[str] — vārdu saraksts (indeksēts kā kokā)
    phases_matrix : (N, 5) — θ_final katram vārdam
    _embedded     : (N, 10) — 10D Eiklīda iegulšana kokam
    _tree         : cKDTree — ātras telpiskās meklēšanas struktūra
    """

    def __init__(
        self,
        words: list[str],
        phases_matrix: np.ndarray,
    ) -> None:
        """
        Inicializē leksikonu un uzbūvē cKDTree.

        Parametri
        ---------
        words         : N vārdi
        phases_matrix : (N, 5) θ_final masīvs
        """
        self.words = words
        self.phases_matrix = phases_matrix  # (N, 5)

        # Pārveido uz 10D Eiklīda telpu
        self._embedded = phases_to_euclidean(phases_matrix)  # (N, 10)

        # Uzbūvē KDTree
        t0 = time.perf_counter()
        self._tree = cKDTree(self._embedded)
        build_ms = (time.perf_counter() - t0) * 1000

        print(f"[FastLexicon] cKDTree uzbūvēts: {len(words)} lapas, "
              f"{self._embedded.shape[1]}D telpa, {build_ms:.2f} ms")

    # ── Ielāde ──────────────────────────────────────────────────

    @classmethod
    def load(cls, filepath: str) -> "FastPhaseLexicon":
        """
        Ielādē masīvus no .npz faila un uzbūvē koku.

        Parametri
        ---------
        filepath : ceļš uz phaseflow_mega_lexicon.npz

        Atgriež
        -------
        FastPhaseLexicon instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Fails nav atrasts: {filepath}\n"
                f"Palaid vispirms: python phaseflow/mass_encoder.py"
            )

        data = np.load(filepath, allow_pickle=True)
        words = list(data["words"])
        phases_matrix = data["phases_matrix"].astype(np.float64)

        print(f"[FastLexicon] Ielādēts: {filepath}")
        print(f"             {len(words)} vārdi, matrica {phases_matrix.shape}")

        return cls(words=words, phases_matrix=phases_matrix)

    # ── Meklēšana ────────────────────────────────────────────────

    def find_closest(
        self,
        target_theta: np.ndarray,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Atrod top_k vārdus, kuru fāžu paraksts ir tuvākais target_theta.

        Parametri
        ---------
        target_theta : (5,) — vaicājuma fāžu vektors (radians)
        top_k        : cik tuvāko vārdu atgriezt

        Atgriež
        -------
        list[(vārds, distance)] — sakārtots no tuvākā uz tālāko
        distance ir Eiklīda distance 10D telpā =
            sqrt(Σ [(cos aᵢ - cos bᵢ)² + (sin aᵢ - sin bᵢ)²])

        Saistība ar cirkulāro distanci:
            D_circ = Σ |e^{iaᵢ} - e^{ibᵢ}|² = Σ 2(1 - cos(aᵢ - bᵢ))
            D_euclid² = D_circ
        """
        target_theta = np.asarray(target_theta, dtype=np.float64).ravel()
        if target_theta.shape[0] != self.phases_matrix.shape[1]:
            raise ValueError(
                f"target_theta dimensija ({target_theta.shape[0]}) != "
                f"leksikona dimensija ({self.phases_matrix.shape[1]})"
            )

        # Pārveido ievadi uz 10D
        query_point = phases_to_euclidean(target_theta)  # (10,)

        # KDTree meklēšana — O(log N)
        distances, indices = self._tree.query(query_point, k=min(top_k, len(self.words)))

        # Skalārs gadījums (top_k=1) → numpy scalar
        if np.ndim(distances) == 0:
            distances = [float(distances)]
            indices = [int(indices)]

        return [(self.words[int(i)], float(d)) for i, d in zip(indices, distances)]

    def get_word_phase(self, word: str) -> "np.ndarray | None":
        """
        Atgriež 5D fāžu vektoru dotajam vārdam, ja tas eksistē leksikonā.

        Parametri
        ---------
        word : vārds (lowercase, bez pieturzīmēm)

        Atgriež
        -------
        np.ndarray (5,) vai None, ja vārds nav atrasts
        """
        try:
            idx = self.words.index(word)
            return self.phases_matrix[idx].copy()
        except ValueError:
            return None

    def find_closest_batch(
        self,
        thetas: np.ndarray,
        top_k: int = 3,
    ) -> list[list[tuple[str, float]]]:
        """
        Batch versija: vienlaicīgi meklē M vaicājumiem.

        Parametri
        ---------
        thetas : (M, 5) — M fāžu vektoru matrica
        top_k  : cik tuvāko vārdu katram vaicājumam

        Atgriež
        -------
        list[list[(vārds, distance)]] — M rezultātu saraksts
        """
        thetas = np.asarray(thetas, dtype=np.float64)
        queries = phases_to_euclidean(thetas)  # (M, 10)

        distances, indices = self._tree.query(queries, k=min(top_k, len(self.words)))

        results = []
        for row_d, row_i in zip(distances, indices):
            results.append(
                [(self.words[int(i)], float(d)) for i, d in zip(row_i, row_d)]
            )
        return results

    # ── Statistika ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.words)

    def __repr__(self) -> str:
        return (
            f"FastPhaseLexicon("
            f"{len(self.words)} vārdi, "
            f"{self._embedded.shape[1]}D kokā)"
        )

    def stats(self) -> None:
        """Izvada statistiku par leksikonu."""
        print(f"\n  FastPhaseLexicon statistika:")
        print(f"  ─────────────────────────────")
        print(f"  Vārdu skaits   : {len(self.words)}")
        print(f"  Oscilatoru sk. : {self.phases_matrix.shape[1]}")
        print(f"  KD-telpa       : {self._embedded.shape[1]}D (cos/sin iegulšana)")
        mean_phase = float(np.mean(self.phases_matrix))
        std_phase = float(np.std(self.phases_matrix))
        print(f"  Vidējā fāze    : {mean_phase:.4f} rad")
        print(f"  Std fāze       : {std_phase:.4f} rad")
        print()


# ═══════════════════════════════════════════════════════════════
#  DEMONSTRĀCIJAS BLOKS
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FastPhaseLexicon demonstrācija")
    parser.add_argument(
        "--npz",
        default="phaseflow_mega_lexicon.npz",
        help="Ceļš uz .npz failu (noklusējums: phaseflow_mega_lexicon.npz)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Cik tuvākos vārdus rādīt (noklusējums: 5)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PhaseFlow — FastPhaseLexicon KDTree Demonstrācija")
    print("=" * 60)

    # 1. Ielādē arhīvu
    lex = FastPhaseLexicon.load(args.npz)
    lex.stats()

    # 2. Ģenerē Kuramoto parakstus demo vārdiem, izmantojot SmartMirror
    print("  Ģenerē vaicājumu fāzes caur SmartMirror ...\n")

    from smart_mirror import SmartMirror
    from core import KuramotoOscillator, BASE_HARMONICS
    from mass_encoder import _compute_signature

    mirror = SmartMirror(verbose=False, enable_learning=False, use_lexicon=False)
    k_matrix = None
    if mirror._persistent_osc is not None:
        k_matrix = mirror._persistent_osc.K_matrix.copy()

    demo_queries = [
        "astoni",       # → BEZGALIBA, OKTAVA?
        "viens",        # → SINGULARITATE, AVOTS?
        "trinpadsmit",  # → ALKIMIJA, FENIKS?
        "harmonija",    # → REZONANSE?
        "iet",          # → KUSTIBA?
        "udens",        # → EZERS, UPITE?
    ]

    print(f"  {'─'*58}")
    print(f"  {'Vaicājums':<16} │ Top-{args.topk} tuvākie vārdi un distances")
    print(f"  {'─'*58}")

    for query in demo_queries:
        theta = _compute_signature(
            query, mirror.encoder, mirror.K, k_matrix, n=len(BASE_HARMONICS)
        )

        t0 = time.perf_counter()
        results = lex.find_closest(theta, top_k=args.topk)
        query_ms = (time.perf_counter() - t0) * 1000

        top_words = " | ".join(
            f"{w} ({d:.3f})" for w, d in results
        )
        print(f"  {query:<16} → {top_words}  [{query_ms:.3f} ms]")

    print(f"  {'─'*58}")

    # 3. Batch meklēšanas ātruma tests
    print(f"\n  Ātruma tests — batch meklēšana ...")
    N_BATCH = 1000
    random_thetas = np.random.uniform(0, 2 * np.pi, size=(N_BATCH, len(BASE_HARMONICS)))

    t0 = time.perf_counter()
    _ = lex.find_closest_batch(random_thetas, top_k=3)
    batch_ms = (time.perf_counter() - t0) * 1000

    print(f"  {N_BATCH} vaicājumi: {batch_ms:.1f} ms kopā "
          f"({batch_ms/N_BATCH*1000:.1f} μs/vaicājums)")
    print(f"\n  FastPhaseLexicon gatavs integrācijai ar SmartMirror!")
    print("=" * 60)
