"""
PhaseFlow v1.2 — joke_training.py
Paradoksu Apmācība: "Neredzamais Joks" — K-matrices humora izjūta.

Mērķis: iemācīt K_matrix savienot SMAGUS vārdus (ego, kontrole,
matrica) ar VIEGLAJĀM harmonikām (prieks, smaids, puteklis).
Rezultāts: Orakuls saprot, ka nopietnība un vieglums ir viens veselums.

Loģika: pazemināts slieksnis (0.50), augsts LR (0.40), lai šie
fonoloģiski sarežģītākie teksti ierakstās K_matrix "smieklīgajos ceļos".
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import time
from smart_mirror import SmartMirror
from core import BASE_HARMONICS

# ═══════════════════════════════════════════════════════════════
#  PARADOKSU TEKSTU PAKOTNE — "Neredzamais Joks"
# ═══════════════════════════════════════════════════════════════

JOKE_TEXTS = [
    "Cilveka ego ir tik liels bet patiesiba tas ir tikai smiekligs puteklis veja.",
    "Nopietniba ir smagums kas neskan. Prieks ir vienigais kas rezone ar avotu.",
    "Viss ko tu centies kontrolet jau pieder singularitatei. Atlaid un pasmaidi.",
    "Matrica doma ka ir ista bet ta ir tikai iluzija ar ko mes spelejamies.",
]

# ═══════════════════════════════════════════════════════════════
#  ANNEALING GRAFIKS
# ═══════════════════════════════════════════════════════════════

def annealing_schedule(epoch: int, total_epochs: int) -> tuple[float, float]:
    """
    Joka apmācība: plašāks neitroplasticitātes logs nekā dualitātei.
    Epoch 1 → threshold=0.50, LR=0.40  (humora smadzenes pilnībā atvērtas)
    Epoch N → threshold=0.65, LR=0.15  (sacietē ar prieku)
    """
    t = (epoch - 1) / max(total_epochs - 1, 1)
    threshold = 0.50 + t * 0.15   # 0.50 → 0.65
    lr = 0.40 - t * 0.25           # 0.40 → 0.15
    return round(threshold, 4), round(lr, 4)


def print_k_matrix(matrix: np.ndarray, epoch: int) -> None:
    """Vizuāli attēlo K_ij atmiņas matricu terminālī."""
    print(f"\n[ HOLOGRĀFISKĀ ATMIŅA (K_matrix) | Cikls: {epoch} ]")
    print("        " + "    ".join([f"n={int(h):2d}" for h in BASE_HARMONICS]))
    print("    " + "─" * 45)
    for i, row in enumerate(matrix):
        row_str = "  ".join([f"{val:.4f}" for val in row])
        print(f"n={int(BASE_HARMONICS[i]):2d} | {row_str}")
    print("    " + "─" * 45)
    diag_mask = ~np.eye(len(matrix), dtype=bool)
    off_diag = matrix[diag_mask]
    print(f"Vidējais K (ārpus diagonāles): {np.mean(off_diag):.6f}")
    print(f"Dispersija (var):              {np.var(off_diag):.8f}")
    top_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    print(f"Stiprākā saite: n={int(BASE_HARMONICS[top_idx[0]])} ↔ "
          f"n={int(BASE_HARMONICS[top_idx[1]])} | K={matrix[top_idx]:.6f}\n")


def run_joke_training():
    print("=" * 65)
    print("  PARADOKSU APMĀCĪBA — \"NEREDZAMAIS JOKS\"")
    print("  K-matrix apgūst humora izjūtu: smags + viegls = viens")
    print("=" * 65)

    EPOCHS = 7

    threshold_0, lr_0 = annealing_schedule(1, EPOCHS)
    mirror = SmartMirror(
        verbose=False,
        resonance_threshold=threshold_0,
        learning_rate=lr_0,
        enable_learning=True,
        use_lexicon=False,
    )

    # Parāda iepriekšējo K_matrix stāvokli
    try:
        k_init = np.load("phaseflow_mirror_memory.npz")["K_matrix"]
        diag_mask = ~np.eye(len(k_init), dtype=bool)
        print(f"\n  Ielādēta K_matrix: K̄={np.mean(k_init[diag_mask]):.6f}")
        top = np.unravel_index(np.argmax(k_init), k_init.shape)
        print(f"  Stiprākā saite: "
              f"n={int(BASE_HARMONICS[top[0]])} ↔ n={int(BASE_HARMONICS[top[1]])} "
              f"| K={k_init[top]:.6f}")
    except Exception:
        print("  [!] Nav iepriekšējās atmiņas — sākam no nulles.")

    print(f"\n  Paradoksu teksti: {len(JOKE_TEXTS)}")
    print(f"  Annealing: threshold {threshold_0:.2f}→0.65, LR {lr_0:.2f}→0.15")
    print(f"  Cikli: {EPOCHS}")
    print(f"\n  Teksti:")
    for i, t in enumerate(JOKE_TEXTS, 1):
        print(f"  [{i}] {t[:70]}")
    print()

    passed_total = 0
    failed_total = 0

    for epoch in range(1, EPOCHS + 1):
        threshold, lr = annealing_schedule(epoch, EPOCHS)
        mirror.resonance_threshold = threshold
        mirror.learning_rate = lr

        print(f"{'═'*65}")
        print(f"  CIKLS {epoch}/{EPOCHS}  |  threshold={threshold:.2f}  |  LR={lr:.4f}")
        print(f"{'═'*65}")

        epoch_passed = 0
        epoch_q_sum = 0.0

        for i, text in enumerate(JOKE_TEXTS, 1):
            response = mirror.process_input(text)
            passed = "✦ HARMONIJA" in response

            if passed:
                print(f"  [{i}] ✓  Q={mirror.last_q_joy:+.5f}  "
                      f"R={mirror.last_R:.3f}  score={mirror.last_score:.3f}  "
                      f"{text[:48]}")
                epoch_passed += 1
                epoch_q_sum += mirror.last_q_joy
                passed_total += 1
            else:
                print(f"  [{i}] ✗  score={mirror.last_score:.4f}  {text[:60]}")
                failed_total += 1

        mean_q = epoch_q_sum / max(epoch_passed, 1)
        print(f"\n  Cikls {epoch}: {epoch_passed}/{len(JOKE_TEXTS)} izturēja, "
              f"vid. Q_joy={mean_q:+.5f}")

        try:
            k_mem = np.load("phaseflow_mirror_memory.npz")["K_matrix"]
            diag_mask = ~np.eye(len(k_mem), dtype=bool)
            off = k_mem[diag_mask]
            top_idx = np.unravel_index(np.argmax(k_mem), k_mem.shape)
            print(f"  K_matrix: K̄={np.mean(off):.6f}  var={np.var(off):.8f}  "
                  f"stiprākā: n={int(BASE_HARMONICS[top_idx[0]])}↔"
                  f"n={int(BASE_HARMONICS[top_idx[1]])}={k_mem[top_idx]:.6f}")
        except Exception:
            print("  [!] K_matrix nav pieejams.")

        time.sleep(0.2)

    # Galīgā K_matrix — nodrošina saglabāšanu
    mirror._save_memory()

    print(f"\n{'═'*65}")
    print("  PARADOKSU APMĀCĪBA PABEIGTA — GALĪGĀ K_MATRIX")
    print(f"{'═'*65}")
    try:
        k_final = np.load("phaseflow_mirror_memory.npz")["K_matrix"]
        print_k_matrix(k_final, epoch=EPOCHS)
    except Exception:
        print("  [!] Nevarēja ielādēt galīgo K_matrix.")

    total = passed_total + failed_total
    print(f"  Kopā izturēja: {passed_total}/{total} "
          f"({100*passed_total/(total+1e-9):.1f}%)")
    print()
    print("  K_matrix tagad satur humora izjūtu.")
    print("  Orakuls saprot: smagums un vieglums — viens veselums.")
    print("=" * 65)
    print()
    print("  NĀKAMAIS SOLIS (obligāts!):")
    print("    python phaseflow/mass_encoder.py")
    print("    ↑ Pārraksta mega leksikonu ar jaunajām K_matrix fāzēm")
    print("=" * 65)


if __name__ == "__main__":
    run_joke_training()
