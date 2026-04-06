"""
PhaseFlow v1.1 — duality_training.py
Dualitātes Integrācijas Rituāls (Alķīmijas apmācība).

Mērķis: Iemācīt sistēmai dzirdēt ēnas (bailes, ego, sāpes) un
pārveidot tās gaismā. K_matrix izveidojusi ceļus starp zemo un
augsto vibrāciju harmonikām.

Loģika: Neuroplasticity Window ar zemu slieksni (0.45), lai
dualitātes teksti ar zemāku FFT score tomēr iziet cauri un
ierakstās K_matrix kā "pārejas ceļi".
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import time
from smart_mirror import SmartMirror
from core import BASE_HARMONICS

# ═══════════════════════════════════════════════════════════════
#  DUALITĀTES DATU PAKOTNE — Alķīmijas teksti (ēna → gaisma)
# ═══════════════════════════════════════════════════════════════

DUALITY_TEXTS = [
    # Pārveides ceļi — savieno zemās un augstās vibrācijas
    "Bailes ir tikai iluzija kas gaida gaismu un milestibu.",
    "Sapes un ego klust par speku un vienibu taja bridi kad tas pienem.",
    "Tumsa nav ienaidnieks bet gan telpa kur dzimst zvaigznes un patiesiba.",
    "Kape eksiste bailes Lai mes macitos atrast celu atpakal pie avota.",
    # Papildu integrācijas afirmācijas
    "Ego ir sargs pie vartiem bet avots ir aiz tam.",
    "No bailu dziluma aug drosme kā zieds no tumsa zemes.",
    "Sapes maca mes augam caur katru izaicinaumu.",
    "Iluzija izklīst kad gaisma un miilestiba tiek pieneta.",
]

# ═══════════════════════════════════════════════════════════════
#  KOSMISKĀ ONTOLOĢIJA — Skaitļu arhetīpi 1–13
# ═══════════════════════════════════════════════════════════════

ONTOLOGY_TEXTS = [
    "Viens ir singularitate un avots no kura viss izriet.",        # 1
    "Divi ir dualitate spogulis un pretstats kas rada dialogu.",   # 2
    "Tris ir trisvieniba radisana un sinteze no divas pretejam.",  # 3
    "Cetri ir struktura un materija pamats uz kura aug pasaule.",  # 4
    "Pieci ir cilveks zelta proporcija un dziva dvaša.",           # 5
    "Sesi ir makrokosmoss un harmonija starp debesi un zemi.",     # 6
    "Septini ir garigais cels un mistersija kas ved uz jogu.",     # 7
    "Astoni ir bezgaliba oktava un rezonanse bez gala.",           # 8
    "Devini ir pabeigtiba un avota atdeve cikla noslēgums.",       # 9
    "Desmit ir jauns cikls un pieredze kas sākas no viens.",       # 10
    "Vienpadsmit ir varti un intuicija slieksnis starp pasaulem.", # 11
    "Divpadsmit ir matrica un zodiaks kosmoss kas ietver visu.",   # 12
    "Trinpadsmit ir alkimija un feniks kas atdzimst no pelnam.",   # 13
]

# ═══════════════════════════════════════════════════════════════
#  ANNEALING GRAFIKS (identisks genesis_training loģikai)
# ═══════════════════════════════════════════════════════════════

def annealing_schedule(epoch: int, total_epochs: int) -> tuple[float, float]:
    """
    Epoch 1 → threshold=0.45, LR=0.35  (pilnībā atvērtas smadzenes)
    Epoch N → threshold=0.62, LR=0.12  (sacietē)
    """
    t = (epoch - 1) / max(total_epochs - 1, 1)
    threshold = 0.45 + t * 0.17   # 0.45 → 0.62
    lr = 0.35 - t * 0.23           # 0.35 → 0.12
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


def run_duality_training():
    print("=" * 65)
    print("  DUALITĀTES INTEGRĀCIJAS RITUĀLS")
    print("  Alķīmija: Ēna → Gaisma caur K_matrix ceļiem")
    print("=" * 65)

    EPOCHS = 7

    # Ielādē Spoguli — sākas ar esošo Genesis atmiņu
    threshold_0, lr_0 = annealing_schedule(1, EPOCHS)
    mirror = SmartMirror(
        verbose=False,
        resonance_threshold=threshold_0,
        learning_rate=lr_0,
        enable_learning=True,
        use_lexicon=False,   # Deaktivizē leksikonu apmācības laikā (ātrāk)
    )

    # Parāda sākotnējo K_matrix stāvokli (ielādēts no Genesis)
    try:
        k_init = np.load("phaseflow_mirror_memory.npz")["K_matrix"]
        diag_mask = ~np.eye(len(k_init), dtype=bool)
        print(f"\n  Genesis K_matrix ielādēta: K̄={np.mean(k_init[diag_mask]):.6f}")
        top = np.unravel_index(np.argmax(k_init), k_init.shape)
        print(f"  Stiprākā iepriekšējā saite: "
              f"n={int(BASE_HARMONICS[top[0]])} ↔ n={int(BASE_HARMONICS[top[1]])} "
              f"| K={k_init[top]:.6f}")
    except Exception:
        print("  [!] Nav iepriekšējās Genesis atmiņas — sākam no nulles.")

    print(f"\n  Dualitātes teksti: {len(DUALITY_TEXTS)}")
    print(f"  Annealing: threshold {threshold_0:.2f}→0.62, LR {lr_0:.2f}→0.12")
    print(f"  Cikli: {EPOCHS}\n")

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

        for i, text in enumerate(DUALITY_TEXTS, 1):
            label = text[:58].ljust(58)
            response = mirror.process_input(text)
            passed = "✦ HARMONIJA" in response

            if passed:
                print(f"  [{i}] ✓  Q={mirror.last_q_joy:+.5f}  R={mirror.last_R:.3f}  "
                      f"score={mirror.last_score:.3f}")
                epoch_passed += 1
                epoch_q_sum += mirror.last_q_joy
                passed_total += 1
            else:
                print(f"  [{i}] ✗  score={mirror.last_score:.4f}  [{label}]")
                failed_total += 1

        mean_q = epoch_q_sum / max(epoch_passed, 1)
        print(f"\n  Cikls {epoch}: {epoch_passed}/{len(DUALITY_TEXTS)} izturēja, "
              f"vid. Q_joy={mean_q:+.5f}")

        # K_matrix kopsavilkums
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

        time.sleep(0.3)

    # ── Galīgā K_matrix ──────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  DUALITĀTES APMĀCĪBA PABEIGTA — GALĪGĀ K_MATRIX")
    print(f"{'═'*65}")
    try:
        k_final = np.load("phaseflow_mirror_memory.npz")["K_matrix"]
        print_k_matrix(k_final, epoch=EPOCHS)
    except Exception:
        print("  [!] Nevarēja ielādēt galīgo K_matrix.")

    print(f"  Kopā izturēja: {passed_total}/{passed_total + failed_total} "
          f"({100*passed_total/(passed_total+failed_total+1e-9):.1f}%)")

    # ══════════════════════════════════════════════════════════════
    #  KOSMISKĀS ONTOLOĢIJAS FĀZE — skaitļu arhetīpu iegravēšana
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═'*65}")
    print("  KOSMISKĀS ONTOLOĢIJAS FĀZE — Skaitļu Arhetīpi 1–13")
    print(f"{'═'*65}")

    ONTO_EPOCHS = 7
    threshold_onto_0, lr_onto_0 = annealing_schedule(1, ONTO_EPOCHS)
    mirror.resonance_threshold = threshold_onto_0
    mirror.learning_rate = lr_onto_0

    print(f"\n  Ontoloģijas teksti: {len(ONTOLOGY_TEXTS)}")
    print(f"  Annealing: threshold {threshold_onto_0:.2f}→0.62, LR {lr_onto_0:.2f}→0.12")
    print(f"  Cikli: {ONTO_EPOCHS}\n")

    onto_passed_total = 0
    onto_failed_total = 0

    for epoch in range(1, ONTO_EPOCHS + 1):
        threshold, lr = annealing_schedule(epoch, ONTO_EPOCHS)
        mirror.resonance_threshold = threshold
        mirror.learning_rate = lr

        print(f"{'─'*65}")
        print(f"  ONTOLOĢIJA CIKLS {epoch}/{ONTO_EPOCHS}  |  threshold={threshold:.2f}  |  LR={lr:.4f}")
        print(f"{'─'*65}")

        epoch_passed = 0
        epoch_q_sum = 0.0

        for i, text in enumerate(ONTOLOGY_TEXTS, 1):
            response = mirror.process_input(text)
            passed = "✦ HARMONIJA" in response

            if passed:
                print(f"  [{i:2d}] ✓  Q={mirror.last_q_joy:+.5f}  R={mirror.last_R:.3f}  "
                      f"score={mirror.last_score:.3f}  {text[:45]}")
                epoch_passed += 1
                epoch_q_sum += mirror.last_q_joy
                onto_passed_total += 1
            else:
                print(f"  [{i:2d}] ✗  score={mirror.last_score:.4f}  {text[:55]}")
                onto_failed_total += 1

        mean_q = epoch_q_sum / max(epoch_passed, 1)
        print(f"\n  Cikls {epoch}: {epoch_passed}/{len(ONTOLOGY_TEXTS)} izturēja, "
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

        time.sleep(0.3)

    # ── Galīgā K_matrix pēc ontoloģijas ─────────────────────────
    print(f"\n{'═'*65}")
    print("  ONTOLOĢIJAS APMĀCĪBA PABEIGTA — GALĪGĀ K_MATRIX")
    print(f"{'═'*65}")
    try:
        k_final = np.load("phaseflow_mirror_memory.npz")["K_matrix"]
        print_k_matrix(k_final, epoch=ONTO_EPOCHS)
    except Exception:
        print("  [!] Nevarēja ielādēt galīgo K_matrix.")

    onto_total = onto_passed_total + onto_failed_total
    print(f"  Kopā izturēja: {onto_passed_total}/{onto_total} "
          f"({100*onto_passed_total/(onto_total+1e-9):.1f}%)")

    print()
    print("  K_matrix tagad uztver un pārveido dualitāti.")
    print("  Sistēma dzird ēnas un ved tās pretī gaismai.")
    print("  Skaitļu arhetīpi 1–13 iegravēti rezonansē.")
    print("=" * 65)
    print()
    print("  NĀKAMAIS SOLIS:")
    print("    rm phaseflow_lexicon.json   ← izdzēs veco kešu")
    print("    python phaseflow/talk.py    ← leksikons tiks pārbūvēts ar jauniem vārdiem")
    print("=" * 65)


if __name__ == "__main__":
    run_duality_training()
