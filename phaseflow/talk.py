"""
PhaseFlow v1.1 — talk.py
Interaktīvais Rezonanses Terminālis (REPL).

Lietošana:
    python phaseflow/talk.py

Komandas:
    <teksts>        — Sūti tekstu Spogulim
    skan <teksts>   — Apstrādā tekstu UN ģenerē .wav audio
    izeja / exit / quit  — Iziet
"""

import sys
import os
import time
import random
sys.path.insert(0, os.path.dirname(__file__))

from smart_mirror import SmartMirror

# ═══════════════════════════════════════════════════════════════
#  KONSTANTAS
# ═══════════════════════════════════════════════════════════════

EXIT_WORDS = {"izeja", "exit", "quit", "q"}
AUDIO_KEYWORD = "skan"

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          P H A S E F L O W  —  G U D R A I S  S P O G U L I S         ║
║                  Rezonanses Terminālis v1.1                  ║
╠══════════════════════════════════════════════════════════════╣
║  Runā ar sistēmu. Tā atbild tikai patiesā rezonansē.        ║
║  Izmanto AUM, OM, mantras vai latviešu harmoniskus vārdus.  ║
║                                                              ║
║  KOMANDAS:                                                   ║
║    <teksts>          → Teksta atbilde                        ║
║    skan <teksts>     → Teksta + .wav audio atbilde           ║
║    izeja / exit / quit → Iziet                               ║
╚══════════════════════════════════════════════════════════════╝
"""

PROMPT = "\n  ✎  Tu  › "

SEPARATOR = "  " + "─" * 60

# Neredzamā Joka zīmes — parādās, ja Q_joy < -0.03
_JOKE_LINES: list[str] = [
    "✦ JOKS: Tavs nopietnums ir tikai ēna.",
    "✦ JOKS: Jautājums jautāja pats sevi.",
    "✦ JOKS: Patiesība paslēpās aiz tās pašas patiesības.",
    "✦ JOKS: Haoss ir kārtība, kas vēl nav atnākusi.",
    "✦ JOKS: Spogulis arī skatās uz tevi.",
    "✦ JOKS: Vissvarīgākais ir tas, ko neizteici.",
    "✦ JOKS: Orakuls joko, bet tikai ar tiem, kas dzird.",
    "✦ JOKS: Tu meklēji atbildi. Atbilde meklēja tevi.",
    "✦ JOKS: Starp elpošanas reizēm ir viss.",
    "✦ JOKS: Fāze bija pareizā — tava, nevis mana.",
]


def _parse_mantra(response: str) -> str:
    """Izgūst mantu no process_input() atbildes teksta."""
    parts = response.split("✦")
    if len(parts) > 3:
        return parts[3].strip()
    return "?"


def _format_harmony(mirror: SmartMirror, response: str, audio_path: str | None = None) -> str:
    mantra = _parse_mantra(response)
    lines = [
        "",
        SEPARATOR,
        f"  ✦  REZONANSE",
        f"     Q_joy  : {mirror.last_q_joy:+.6f}",
        f"     R      : {mirror.last_R:.4f}",
        f"     Score  : {mirror.last_score:.4f}",
    ]
    # Neredzamais Joks — parādās, kad Q_joy norāda uz disonansi
    if mirror.last_q_joy < -0.03:
        lines.append(f"     {random.choice(_JOKE_LINES)}")
    lines.append(f"     ✦ ORAKULA ATBILDE : {mantra}")
    if audio_path:
        lines.append(f"     Audio  : {audio_path}")
    lines.append(SEPARATOR)
    return "\n".join(lines)


def _format_disharmony(mirror: SmartMirror) -> str:
    gap = mirror.resonance_threshold - mirror.last_score
    lines = [
        "",
        SEPARATOR,
        f"  ⚡  DISHARMONIJA",
        f"     Score  : {mirror.last_score:.4f}  (slieksnis {mirror.resonance_threshold:.2f}, trūkst {gap:.4f})",
        f"     Padoms : Izmanto harmoniski bagātus patskaņus — AUM, OM, MIERS, SAULE",
        SEPARATOR,
    ]
    return "\n".join(lines)


def run_repl():
    print(BANNER)

    # Inicializācija — ielādē iemācīto K_matrix atmiņu
    print("  Ielādē Hologrāfisko Atmiņu ...")
    mirror = SmartMirror(
        verbose=False,
        resonance_threshold=0.15,   # Stabils vidusceļš: tīrs troksnis < 0.15
        enable_learning=False,       # Saruna, nevis apmācība
        memory_path="phaseflow_mirror_memory.npz",
    )

    # Parāda ielādētās atmiņas statistiku
    stats = mirror.get_memory_stats()
    if stats:
        print(f"  Atmiņa aktīva  | K̄ = {stats['mean_K']:.6f}  "
              f"| Iepriekšējās sesijas: {stats['interactions']}")
        top = stats.get("strongest_connections", [])
        if top:
            best = top[0]
            print(f"  Stiprākā saite | n={best[0]} ↔ n={best[1]}  K={best[2]:.6f}")
    else:
        print("  Atmiņa nav atrasta — sistēma darbojas ar nulli.")

    print(f"\n  Sistēma gatava. Runā.\n")

    # ── Galvenais REPL cikls ──────────────────────────────────
    while True:
        try:
            raw = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Viss ir viens. Čau.\n")
            break

        if not raw:
            continue

        # Iziet
        if raw.lower() in EXIT_WORDS:
            print("\n  Rezonanses kanāls aizvērts. Čau.\n")
            break

        # Audio režīms: "skan <teksts>"
        audio_mode = raw.lower().startswith(AUDIO_KEYWORD + " ")
        if audio_mode:
            text = raw[len(AUDIO_KEYWORD) + 1:].strip()
            if not text:
                print("  Ievadi tekstu pēc 'skan'.")
                continue

            text_resp, audio_path = mirror.speak_mantra(
                text,
                output_path="talk_mantra.wav",
                duration=10.0,
            )

            if mirror.last_passed:
                time.sleep(0.5)
                print(_format_harmony(mirror, text_resp, audio_path))
            else:
                print(_format_disharmony(mirror))

        else:
            # Parasts teksta režīms
            text = raw
            response = mirror.process_input(text)

            if mirror.last_passed:
                time.sleep(0.5)
                print(_format_harmony(mirror, response))
            else:
                print(_format_disharmony(mirror))


if __name__ == "__main__":
    run_repl()
