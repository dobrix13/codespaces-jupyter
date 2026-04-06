"""
PhaseFlow v1.4 — manuscript_immersion.py
Dziļā Iegrimšana: manuskripts → K-matricas apmācība

Plūsma:
  1. Ielādē teksta failu, sadala "jēgas vienībās"
  2. Katru vienību nodod SmartMirror (enable_learning=True)
  3. Hebbiana mācīšanās atjaunina K-matricu pēc katras vienības
  4. Noguruma mehānisms samazina learning_rate ar laiku
  5. Jaunos vārdus automātiski iekodē un pievieno leksikonam
  6. Saglabā atjaunināto phaseflow_mirror_memory.npz un leksikonu

Izmantošana:
    python phaseflow/manuscript_immersion.py --text manuscript.txt
    python phaseflow/manuscript_immersion.py --text neredzamais_joks.txt --epochs 3
    python phaseflow/manuscript_immersion.py --text teksts.txt --dry-run
"""

from __future__ import annotations

import re
import sys
import os
import math
import time
import argparse
import unicodedata

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from smart_mirror import SmartMirror
from mass_encoder import _compute_signature, OUTPUT_PATH


# ═══════════════════════════════════════════════════════════════
#  PARAMETRI
# ═══════════════════════════════════════════════════════════════

IMMERSION_THRESHOLD   = 0.10   # Ļoti zems — uzsūc visu
INITIAL_LEARNING_RATE = 0.15   # Sākotnējais mācīšanās ātrums
FATIGUE_HALFLIFE      = 0.35   # Pēc 35% teksta ātrums samazinās 2x
MIN_LEARNING_RATE     = 0.02   # Apakšējā robeža (neapstājas pilnīgi)
MIN_UNIT_LEN          = 15     # Minimālais jēgas vienības garums (burti)
MAX_UNIT_LEN          = 600    # Maksimālais vienības garums pirms dalīšanas

# Latviešu burti (atkārtots šeit, lai neiemportētu lexicon_cleaner)
_LV_CHARS = r"a-zA-ZāčēģīķļņšūžĀČĒĢĪĶĻŅŠŪŽ"
_WORD_RE  = re.compile(rf"[{_LV_CHARS}]{{2,}}")


# ═══════════════════════════════════════════════════════════════
#  1. TEKSTA APSTRĀDE
# ═══════════════════════════════════════════════════════════════

def load_text(filepath: str) -> str:
    """Ielādē teksta failu, normalizē rindas un kodējumu."""
    with open(filepath, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    # Normalizē Unicode (NFC) un rindas beigas
    raw = unicodedata.normalize("NFC", raw)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    return raw


def split_into_units(text: str) -> list[str]:
    """
    Sadala tekstu "jēgas vienībās" (rindkopās / teikumos).

    Hierarhija:
    1. Sadala pa dubultrindkopām (tukša rinda starp blokiem)
    2. Pārāk garus blokus sadala pa teikumiem (., !, ?)
    3. Atmet īsas / tukšas vienības

    Pieturzīmes tiek saglabātas — tās ir fāžu pārejas punkti.
    """
    # Vairākas tukšas rindas → viena atdalītāja rinda
    text = re.sub(r"\n{2,}", "\n\n", text)
    paragraphs = text.split("\n\n")

    units: list[str] = []
    for para in paragraphs:
        # Sakārt liekās atstarpes iekšienē, saglabā punktus
        cleaned = re.sub(r"[ \t]+", " ", para.strip())
        if not cleaned:
            continue

        if len(cleaned) <= MAX_UNIT_LEN:
            units.append(cleaned)
        else:
            # Sadala pa teikumiem: ., !, ? + atstarpe vai rindas beigas
            sentences = re.split(r"(?<=[.!?])\s+", cleaned)
            buf = ""
            for sent in sentences:
                if len(buf) + len(sent) + 1 <= MAX_UNIT_LEN:
                    buf = (buf + " " + sent).strip() if buf else sent
                else:
                    if buf:
                        units.append(buf)
                    buf = sent
            if buf:
                units.append(buf)

    # Filtrē pārāk īsas vienības
    return [u for u in units if len(u) >= MIN_UNIT_LEN]


def extract_words(text: str) -> list[str]:
    """Iegūst unikālus, attīrītus vārdus no teksta."""
    tokens = _WORD_RE.findall(text.lower())
    seen: set[str] = set()
    result: list[str] = []
    for t in tokens:
        t = unicodedata.normalize("NFC", t)
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ═══════════════════════════════════════════════════════════════
#  2. NOGURUMA MEHĀNISMS
# ═══════════════════════════════════════════════════════════════

def fatigue_lr(
    step: int,
    total_steps: int,
    initial_lr: float = INITIAL_LEARNING_RATE,
    halflife: float = FATIGUE_HALFLIFE,
    min_lr: float = MIN_LEARNING_RATE,
) -> float:
    """
    Aprēķina "noguruša" mācīšanās ātrumu.

    Eksponenciāla samazināšanās:
        lr(t) = max(min_lr, initial_lr · 2^{-t / (halflife · total)})

    Parametri
    ---------
    step       : pašreizējais solis (0-indeksēts)
    total_steps: kopējais soļu skaits
    halflife   : frakcija [0,1] — pēc cik % teksta ātrums puse
    """
    if total_steps <= 0:
        return initial_lr
    t = step / total_steps
    decay = math.pow(2.0, -t / halflife)
    return max(min_lr, initial_lr * decay)


# ═══════════════════════════════════════════════════════════════
#  3. LEKSIKONA PAPILDINĀŠANA
# ═══════════════════════════════════════════════════════════════

def encode_new_words(
    words: list[str],
    mirror: SmartMirror,
    lexicon_path: str = OUTPUT_PATH,
) -> int:
    """
    Iekodē jaunus vārdus un pievieno tos leksikonam (.npz).

    Atgriež pievienoto vārdu skaitu.
    """
    if not words:
        return 0

    # Ielādē esošo leksikonu
    data = np.load(lexicon_path, allow_pickle=True)
    existing_words: list[str] = list(data["words"])
    existing_phases: np.ndarray = data["phases_matrix"].astype(np.float64)
    n_osc = existing_phases.shape[1]

    existing_set = set(existing_words)
    truly_new = [w for w in words if w not in existing_set]

    if not truly_new:
        return 0

    k_matrix = None
    if mirror._persistent_osc is not None:
        k_matrix = mirror._persistent_osc.K_matrix.copy()

    print(f"\n  [Leksikons] Iekodē {len(truly_new)} jaunus vārdus ...")
    new_phases: list[np.ndarray] = []
    for i, w in enumerate(truly_new, 1):
        ph = _compute_signature(w, mirror.encoder, mirror.K, k_matrix, n=n_osc)
        new_phases.append(ph)
        if i % 50 == 0 or i == len(truly_new):
            print(f"  [Leksikons] {i}/{len(truly_new)} ...", end="\r")
    print()

    # Apvieno un saglabā
    all_words = existing_words + truly_new
    all_phases = np.vstack([existing_phases, np.array(new_phases)])

    np.savez_compressed(
        lexicon_path,
        words=np.array(all_words, dtype=object),
        phases_matrix=all_phases.astype(np.float32),
    )
    print(f"  [Leksikons] Saglabāts: {lexicon_path}  "
          f"({len(existing_words)} + {len(truly_new)} = {len(all_words)} vārdi)")

    return len(truly_new)


# ═══════════════════════════════════════════════════════════════
#  4. IEGRIMŠANAS CILPA
# ═══════════════════════════════════════════════════════════════

def run_immersion(
    units: list[str],
    mirror: SmartMirror,
    epoch: int = 1,
    total_epochs: int = 1,
    dry_run: bool = False,
) -> dict:
    """
    Galvenā iegrimšanas cilpa — nodod katru vienību SmartMirror.

    Atgriež statistiku: {passed, failed, avg_score, avg_q_joy}
    """
    n = len(units)
    # Solis ir (epoch-1)*n + i, kopējais soļu skaits = total_epochs * n
    total_steps = total_epochs * n
    step_offset = (epoch - 1) * n

    stats = {"passed": 0, "failed": 0, "scores": [], "q_joys": [], "lr_values": []}

    print(f"\n  {'─'*60}")
    print(f"  IEGRIMŠANA — Epoha {epoch}/{total_epochs}  |  {n} vienības")
    print(f"  Sākotnējais LR: {fatigue_lr(step_offset, total_steps):.4f}"
          f"  →  Beigu LR: {fatigue_lr(step_offset + n - 1, total_steps):.4f}")
    print(f"  {'─'*60}\n")

    t_start = time.time()

    for i, unit in enumerate(units):
        global_step = step_offset + i
        lr = fatigue_lr(global_step, total_steps)

        # Iestati learning_rate pirms process_input
        mirror.learning_rate = lr
        stats["lr_values"].append(lr)

        if not dry_run:
            _ = mirror.process_input(unit)
        else:
            # Dry-run: tikai rezonanses pārbaude, bez mācīšanās
            mirror._check_resonance(unit)

        # Statistika
        if mirror.last_passed:
            stats["passed"] += 1
        else:
            stats["failed"] += 1

        stats["scores"].append(mirror.last_score)
        stats["q_joys"].append(mirror.last_q_joy)

        # Progresa indikators (~katrs 5. %)
        report_every = max(1, n // 20)
        if (i + 1) % report_every == 0 or (i + 1) == n:
            elapsed = time.time() - t_start
            pct = 100.0 * (i + 1) / n
            bar_w = 30
            filled = int(pct / 100 * bar_w)
            bar = "█" * filled + "░" * (bar_w - filled)
            eta = elapsed / (i + 1) * (n - i - 1)
            status = "✓" if mirror.last_passed else "·"
            print(
                f"  [{bar}] {pct:5.1f}%  "
                f"LR={lr:.4f}  "
                f"Score={mirror.last_score:.3f} {status}  "
                f"ETA:{eta:4.0f}s",
                end="\r",
            )

    elapsed_total = time.time() - t_start
    print()  # rindas pārvietošana pēc \r

    avg_score  = float(np.mean(stats["scores"])) if stats["scores"] else 0.0
    avg_q_joy  = float(np.mean(stats["q_joys"])) if stats["q_joys"] else 0.0
    pass_rate  = stats["passed"] / n * 100 if n else 0.0

    print(f"\n  Epoha {epoch} pabeigta  ({elapsed_total:.1f}s)")
    print(f"  Caurlaidība : {pass_rate:.1f}%  ({stats['passed']}/{n})")
    print(f"  Vid. Score  : {avg_score:.4f}")
    print(f"  Vid. Q_joy  : {avg_q_joy:+.4f}")

    if mirror._persistent_osc is not None and not dry_run:
        mean_k = mirror._persistent_osc.coupling_strength()
        print(f"  K̄ pēc epohas: {mean_k:.6f}")

    stats["avg_score"] = avg_score
    stats["avg_q_joy"] = avg_q_joy
    stats["pass_rate"] = pass_rate
    return stats


# ═══════════════════════════════════════════════════════════════
#  5. GALVENĀ IEEJAS PUNKTA FUNKCIJA
# ═══════════════════════════════════════════════════════════════

def immerse(
    text_path: str,
    lexicon_path: str = OUTPUT_PATH,
    memory_path: str = "phaseflow_mirror_memory.npz",
    epochs: int = 1,
    dry_run: bool = False,
    quiet: bool = False,
) -> None:
    """
    Pilna iegrimšanas plūsma.

    Parametri
    ---------
    text_path    : ceļš uz .txt manuskriptu
    lexicon_path : phaseflow_mega_lexicon.npz
    memory_path  : phaseflow_mirror_memory.npz
    epochs       : cik reizes pāriet cauri tekstam
    dry_run      : neko nesaglabā, tikai statistika
    quiet        : mazāk teksta izvade
    """
    print(f"\n{'═'*62}")
    print(f"  P H A S E F L O W  —  D Z I Ļ Ā  I E G R I M Š A N A")
    print(f"{'═'*62}")
    print(f"  Teksts    : {text_path}")
    print(f"  Leksikons : {lexicon_path}")
    print(f"  Atmiņa    : {memory_path}")
    print(f"  Epohas    : {epochs}")
    print(f"  Dry-run   : {dry_run}")
    print(f"{'═'*62}\n")

    # ── Ielādē tekstu ────────────────────────────────────────
    if not os.path.exists(text_path):
        print(f"[!] Fails nav atrasts: {text_path}")
        sys.exit(1)

    raw_text = load_text(text_path)
    units = split_into_units(raw_text)
    print(f"  Teksts ielādēts: {len(raw_text):,} burti  →  {len(units)} jēgas vienības")

    # ── Iegūst visus vārdus no manuskripta ───────────────────
    all_manuscript_words = extract_words(raw_text)
    print(f"  Unikāli vārdi manuskriptā: {len(all_manuscript_words)}")

    # ── Inicializē SmartMirror ar mācīšanās režīmu ──────────
    print(f"\n  Inicializē SmartMirror (mācīšanās ieslēgta) ...")
    mirror = SmartMirror(
        verbose=False,
        resonance_threshold=IMMERSION_THRESHOLD,
        enable_learning=True,
        learning_rate=INITIAL_LEARNING_RATE,
        memory_path=memory_path,
        lexicon_path=lexicon_path,
    )

    if mirror._persistent_osc is not None:
        mean_k = mirror._persistent_osc.coupling_strength()
        history_n = len(mirror._persistent_osc.learning_history)
        print(f"  Atmiņa ielādēta | K̄={mean_k:.6f} | Sesijas: {history_n}")
    else:
        print("  Atmiņa nav atrasta — sākas no nulles.")

    # ── Noskaidro jaunos vārdus pirms iegrimšanas ───────────
    new_words: list[str] = []
    if mirror._lexicon is not None:
        known: set[str] = set(mirror._lexicon.words)
        new_words = [w for w in all_manuscript_words if w not in known]
        print(f"  Jauni vārdi leksikonam: {len(new_words)}")
    else:
        print("  [!] Leksikons nav ielādēts — jaunie vārdi netiks pievienoti.")

    # ── Iegrimšanas cilpa ────────────────────────────────────
    all_stats: list[dict] = []
    for epoch in range(1, epochs + 1):
        stats = run_immersion(
            units,
            mirror,
            epoch=epoch,
            total_epochs=epochs,
            dry_run=dry_run,
        )
        all_stats.append(stats)

    # ── Kopsavilkums par visām epohām ────────────────────────
    if epochs > 1 and not quiet:
        print(f"\n  {'─'*60}")
        print(f"  KOPSAVILKUMS ({epochs} epohas)")
        print(f"  {'─'*60}")
        for e, s in enumerate(all_stats, 1):
            print(f"  Epoha {e}: caurlaidība={s['pass_rate']:.1f}%  "
                  f"vid.score={s['avg_score']:.4f}  "
                  f"vid.Q_joy={s['avg_q_joy']:+.4f}")

    # ── Saglabā atmiņu ───────────────────────────────────────
    if not dry_run and mirror._persistent_osc is not None:
        try:
            mirror._persistent_osc.save_memory(memory_path)
            print(f"\n  ✓ Atmiņa saglabāta: {memory_path}")
        except Exception as exc:
            print(f"\n  [!] Nevarēja saglabāt atmiņu: {exc}")

    # ── Iekodē un pievieno jaunos vārdus leksikonam ─────────
    if new_words and not dry_run and os.path.exists(lexicon_path):
        added = encode_new_words(new_words, mirror, lexicon_path=lexicon_path)
        if added > 0:
            print(f"  ✓ Leksikons papildināts ar {added} jauniem vārdiem: {lexicon_path}")
    elif dry_run:
        print(f"\n  Dry-run — atmiņa un leksikons NAV mainīti.")

    # ── Galīgā K-matricas statistika ─────────────────────────
    if mirror._persistent_osc is not None:
        osc = mirror._persistent_osc
        mean_k = osc.coupling_strength()
        top = osc.strongest_connections(top_n=3)
        print(f"\n  {'─'*60}")
        print(f"  GALĪGĀ K-MATRICA")
        print(f"  K̄ = {mean_k:.6f}")
        for n0, n1, k_val in top:
            print(f"    Stiprākā saite: n={n0} ↔ n={n1}  K={k_val:.6f}")
        print(f"  {'─'*60}")

    print(f"\n  Iegrimšana pabeigta. Orakuls ir mācījies.\n")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PhaseFlow Dziļā Iegrimšana — manuskripts → K-matricas apmācība"
    )
    parser.add_argument(
        "--text", required=True,
        help="Ceļš uz .txt manuskriptu (UTF-8)"
    )
    parser.add_argument(
        "--lexicon", default=OUTPUT_PATH,
        help=f"Leksikona .npz fails (noklusēts: {OUTPUT_PATH})"
    )
    parser.add_argument(
        "--memory", default="phaseflow_mirror_memory.npz",
        help="Atmiņas .npz fails (noklusēts: phaseflow_mirror_memory.npz)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Cik reizes pāriet cauri tekstam (noklusēts: 1)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Tikai statistika — neko nesaglabā"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Mazāk izvades teksta"
    )
    args = parser.parse_args()

    immerse(
        text_path=args.text,
        lexicon_path=args.lexicon,
        memory_path=args.memory,
        epochs=args.epochs,
        dry_run=args.dry_run,
        quiet=args.quiet,
    )
