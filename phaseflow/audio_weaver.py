"""
PhaseFlow v1.0 — audio_weaver.py
Skaņas Sintezators: Fāžu Haoss → Harmoniskais Akords

Pārveido Kuramoto oscilatoru dinamiku dzirdamā .wav failā:
  - Sākums (UV): disharmonisks "kosmiskais vējš"
  - Beigas (IR): dzidrs, varens MANTRA akords

Solfedžo Frekvences:
  n=1  → 174 Hz (Zeme / Pamats)
  n=3  → 285 Hz (Kvantiskais lauks)
  n=6  → 528 Hz (Transformācija / Q_joy)
  n=9  → 852 Hz (Intuīcija)
  n=11 → 963 Hz (Singularitāte / Crown)
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Sequence
from pathlib import Path

import numpy as np

# ── Importē core.py ─────────────────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from core import (
    PHI, PSI_42, LAMBDA_21, BASE_HARMONICS,
    KuramotoOscillator, q_joy,
)


# ═══════════════════════════════════════════════════════════════════
# 1. SOLFEDŽO FREKVENCES — HARMONIKU MAPĒJUMS
# ═══════════════════════════════════════════════════════════════════

# Bāzes harmonikas → Solfedžo frekvences (Hz)
SOLFEGGIO_MAP: dict[float, float] = {
    1.0:  174.0,   # Zeme / Pamats
    3.0:  285.0,   # Kvantiskais lauks
    6.0:  528.0,   # Transformācija / Q_joy (Mīlestības frekvence)
    9.0:  852.0,   # Intuīcija / Trešā acs
    11.0: 963.0,   # Singularitāte / Crown čakra
}

# Noklusējuma sample rate
DEFAULT_SAMPLE_RATE: int = 44100


# ═══════════════════════════════════════════════════════════════════
# 2. AUDIO WEAVER — GALVENĀ KLASE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AudioWeaver:
    """
    Sintezē audio no Kuramoto oscilatoru fāžu dinamikas.

    Parametri
    ---------
    sample_rate   : audio sample rate (Hz)
    duration      : kopējais audio garums (sekundes)
    heartbeat_bpm : sirdspukstu ātrums amplitūdas modulācijai
    fade_in       : ievadīšanas laiks (sekundes)
    fade_out      : izvadīšanas laiks (sekundes)
    master_volume : kopējais skaļums [0, 1]
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    duration: float = 12.0
    heartbeat_bpm: float = 60.0
    fade_in: float = 0.5
    fade_out: float = 1.5
    master_volume: float = 0.7

    def _time_array(self) -> np.ndarray:
        """Rada laika masīvu visam audio garumam."""
        n_samples = int(self.sample_rate * self.duration)
        return np.linspace(0, self.duration, n_samples, endpoint=False)

    def _heartbeat_envelope(self, t: np.ndarray) -> np.ndarray:
        """
        Amplitūdas modulācija — "sirds ritms" (60 BPM = 1 Hz).

        Izmanto mīkstu sinusoīdu: 0.7 + 0.3 * sin(2π * f_heart * t)
        Tas rada pulsējošu, bet ne pilnīgi klusu efektu.
        """
        f_heart = self.heartbeat_bpm / 60.0  # Hz
        # Pulsācija starp 0.5 un 1.0 (nav pilnīgs klusums)
        return 0.75 + 0.25 * np.sin(2 * np.pi * f_heart * t - np.pi / 2)

    def _fade_envelope(self, t: np.ndarray) -> np.ndarray:
        """Fade-in/fade-out aploksne."""
        envelope = np.ones_like(t)

        # Fade in
        fade_in_samples = int(self.fade_in * self.sample_rate)
        if fade_in_samples > 0:
            envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)

        # Fade out
        fade_out_samples = int(self.fade_out * self.sample_rate)
        if fade_out_samples > 0:
            envelope[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)

        return envelope

    def _synchronization_curve(self,
                               t: np.ndarray,
                               R_history: np.ndarray | None = None) -> np.ndarray:
        """
        Sinhronizācijas līkne R(t) — cik lielā mērā fāzes ir saskaņotas.

        Ja R_history nav dots, simulē tipisku Kuramoto pāreju:
        R(t) = tanh(α * t / T) — S-veida līkne no 0 uz ~1
        """
        if R_history is not None and len(R_history) > 1:
            # Interpolē R_history uz audio laika skalu
            t_sim = np.linspace(0, self.duration, len(R_history))
            return np.interp(t, t_sim, R_history)

        # Noklusējuma S-veida līkne
        midpoint = self.duration * 0.4
        steepness = 6.0 / self.duration
        return (1 + np.tanh(steepness * (t - midpoint))) / 2

    def synthesize(self,
                   theta_history: np.ndarray | None = None,
                   R_history: np.ndarray | None = None,
                   init_phases: np.ndarray | None = None) -> np.ndarray:
        """
        Sintezē audio signālu no fāžu dinamikas.

        Parametri
        ---------
        theta_history : (T, N) masīvs ar oscilatoru fāzēm katrā laika solī
        R_history     : (T,) masīvs ar kārtības parametru R(t)
        init_phases   : (N,) sākotnējās fāzes, ja theta_history nav dots

        Returns
        -------
        np.ndarray : mono audio signāls [-1, 1]
        """
        t = self._time_array()
        n_samples = len(t)
        n_harmonics = len(BASE_HARMONICS)

        # ── Sinhronizācijas līkne ─────────────────────────
        R_t = self._synchronization_curve(t, R_history)

        # ── Fāžu interpolācija vai ģenerēšana ─────────────
        if theta_history is not None and len(theta_history) > 1:
            # Interpolē no simulācijas uz audio laiku
            T_sim = len(theta_history)
            t_sim = np.linspace(0, self.duration, T_sim)
            phases = np.zeros((n_samples, n_harmonics))
            for i in range(n_harmonics):
                phases[:, i] = np.interp(t, t_sim, theta_history[:, i])
        else:
            # Ģenerē fāzes: sākumā izkliedētas, beigās sinhronizētas
            if init_phases is None:
                rng = np.random.default_rng(42)
                init_phases = rng.uniform(0, 2 * np.pi, n_harmonics)

            # Mērķa fāze — visas konverģē uz PSI_42
            target_phase = PSI_42

            phases = np.zeros((n_samples, n_harmonics))
            for i in range(n_harmonics):
                # Interpolācija: init_phase → target_phase, svērtā ar R(t)
                phase_diff = (target_phase - init_phases[i]) % (2 * np.pi)
                if phase_diff > np.pi:
                    phase_diff -= 2 * np.pi
                phases[:, i] = init_phases[i] + R_t * phase_diff

        # ── Audio sintēze ─────────────────────────────────
        audio = np.zeros(n_samples, dtype=np.float64)

        for i, n in enumerate(BASE_HARMONICS):
            freq = SOLFEGGIO_MAP.get(n, 440.0)

            # Fāze kustības laikā
            phase_t = phases[:, i]

            # Pamata sinusoīda ar kustīgu fāzi
            wave = np.sin(2 * np.pi * freq * t + phase_t)

            # Pievienojam otro harmoniku (oktāva) blendā ar R(t)
            # Kad R aug, otrā harmonika kļūst spēcīgāka → bagātāka skaņa
            overtone = 0.3 * R_t * np.sin(4 * np.pi * freq * t + phase_t * 2)

            # Amplitūda: augstākas harmonikas klusākas (1/Φ^i svēršana)
            amplitude = 1.0 / (PHI ** (i * 0.5))

            audio += amplitude * (wave + overtone)

        # ── Normalizācija ─────────────────────────────────
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp

        # ── Aploksnes ─────────────────────────────────────
        heartbeat = self._heartbeat_envelope(t)
        fade = self._fade_envelope(t)
        envelope = heartbeat * fade

        audio = audio * envelope * self.master_volume

        return audio

    def synthesize_from_oscillator(self,
                                   oscillator: KuramotoOscillator) -> np.ndarray:
        """
        Sintezē audio tieši no KuramotoOscillator objekta.

        Piemērs
        -------
        >>> osc = KuramotoOscillator(K=2.5, seed=42)
        >>> osc.run(steps=500)
        >>> weaver = AudioWeaver(duration=10.0)
        >>> audio = weaver.synthesize_from_oscillator(osc)
        """
        history = np.array(oscillator.history)

        # Aprēķina R(t) no vēstures
        z_mean = np.mean(np.exp(1j * history), axis=1)
        R_history = np.abs(z_mean)

        return self.synthesize(theta_history=history, R_history=R_history)

    def save_wav(self,
                 audio: np.ndarray,
                 filepath: str | Path = "mantra_resonance.wav") -> Path:
        """
        Saglabā audio kā .wav failu (16-bit PCM).

        Returns
        -------
        Path : pilns ceļš uz saglabāto failu
        """
        try:
            from scipy.io import wavfile
        except ImportError:
            raise ImportError(
                "scipy nav instalēts. Palaidiet: pip install scipy"
            )

        filepath = Path(filepath)

        # Konvertē uz 16-bit integer
        audio_int16 = np.int16(audio * 32767)

        wavfile.write(str(filepath), self.sample_rate, audio_int16)

        return filepath.resolve()


# ═══════════════════════════════════════════════════════════════════
# 3. ĒRTAS FUNKCIJAS
# ═══════════════════════════════════════════════════════════════════

def weave_mantra(text: str | None = None,
                 duration: float = 12.0,
                 output_path: str = "mantra_resonance.wav",
                 play: bool = False) -> tuple[np.ndarray, Path]:
    """
    Pilna plūsma: teksts → Kuramoto sinhronizācija → audio → .wav

    Parametri
    ---------
    text        : ievades teksts (ja None, izmanto noklusējuma "AUM")
    duration    : audio garums sekundēs
    output_path : izvades .wav faila ceļš
    play        : vai mēģināt atskaņot audio (prasa papildu bibliotēkas)

    Returns
    -------
    (audio, path) : sintezētais audio masīvs un ceļš uz .wav failu
    """
    from smart_mirror import TextPhaseEncoder

    # ── Teksta kodēšana → sākotnējās fāzes ────────────────
    if text is None:
        text = "AUM"

    encoder = TextPhaseEncoder()
    init_phases = encoder.text_to_init_phases(text, n_oscillators=len(BASE_HARMONICS))

    # ── Kuramoto simulācija ───────────────────────────────
    osc = KuramotoOscillator(
        n_oscillators=len(BASE_HARMONICS),
        K=3.0,
        theta0=init_phases,
        seed=None,
    )

    # Skrien simulāciju proporcionāli audio garumam
    steps = int(duration * 50)  # ~50 soļi/sekundē
    osc.run(steps=steps, dt=0.02)

    # ── Audio sintēze ─────────────────────────────────────
    weaver = AudioWeaver(duration=duration)
    audio = weaver.synthesize_from_oscillator(osc)

    # ── Saglabāšana ───────────────────────────────────────
    path = weaver.save_wav(audio, output_path)

    print(f"[AudioWeaver] Audio saglabāts: {path}")
    print(f"              Garums: {duration:.1f}s @ {weaver.sample_rate} Hz")
    print(f"              Teksts: '{text}'")

    # ── Atskaņošana (ja iespējams) ────────────────────────
    if play:
        _try_play(path)

    return audio, path


def _try_play(filepath: Path) -> None:
    """Mēģina atskaņot audio failu."""
    import subprocess
    import shutil

    # Mēģina atrast audio atskaņotāju
    players = ["aplay", "paplay", "afplay", "mpv", "ffplay"]

    for player in players:
        if shutil.which(player):
            try:
                print(f"[AudioWeaver] Atskaņo ar {player}...")
                subprocess.run([player, str(filepath)],
                               capture_output=True, timeout=60)
                return
            except Exception:
                continue

    print("[AudioWeaver] Nav pieejams audio atskaņotājs. Failu var atvērt manuāli.")


def create_chakra_drone(duration: float = 30.0,
                        output_path: str = "chakra_drone.wav") -> Path:
    """
    Rada ilgstošu "drone" skaņu ar visām 5 Solfedžo frekvencēm.

    Šī nav Kuramoto dinamika — vienkārši statisks harmonisks akords
    meditatīviem nolūkiem.
    """
    weaver = AudioWeaver(duration=duration)
    t = weaver._time_array()
    n_samples = len(t)

    audio = np.zeros(n_samples, dtype=np.float64)

    for i, (n, freq) in enumerate(SOLFEGGIO_MAP.items()):
        # Statiska fāze (nav kustības)
        phase = PSI_42 + i * PHI

        # Pamata sinusoīda
        wave = np.sin(2 * np.pi * freq * t + phase)

        # Viegla vibrato (LFO) — 0.1 Hz
        vibrato = 1.0 + 0.02 * np.sin(2 * np.pi * 0.1 * t + i)

        # Amplitūda
        amplitude = 1.0 / (PHI ** (i * 0.3))

        audio += amplitude * wave * vibrato

    # Normalizācija un aploksnes
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp

    envelope = weaver._heartbeat_envelope(t) * weaver._fade_envelope(t)
    audio = audio * envelope * weaver.master_volume

    path = weaver.save_wav(audio, output_path)
    print(f"[AudioWeaver] Čakru drone saglabāts: {path} ({duration:.0f}s)")

    return path


# ═══════════════════════════════════════════════════════════════════
# __MAIN__ — DEMONSTRĀCIJA
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PhaseFlow v1.0 — Audio Weaver")
    print("  Fāžu Haoss → Harmoniskais Akords")
    print("=" * 60)
    print()

    # ── Solfedžo frekvences info ──────────────────────────
    print("Solfedžo frekvences:")
    print("-" * 40)
    for n, freq in SOLFEGGIO_MAP.items():
        names = {
            1.0: "Zeme / Pamats",
            3.0: "Kvantiskais lauks",
            6.0: "Transformācija / Q_joy",
            9.0: "Intuīcija",
            11.0: "Singularitāte / Crown",
        }
        print(f"  n={n:2.0f} → {freq:4.0f} Hz  ({names.get(n, '')})")
    print()

    # ── Test 1: Kuramoto simulācija → audio ───────────────
    print("▸ Test 1: Kuramoto sinhronizācija → audio")
    print("-" * 40)

    osc = KuramotoOscillator(K=3.0, seed=42)
    print(f"  Sākums: R = {osc.order_parameter()[0]:.4f}")

    history = osc.run(steps=600, dt=0.02)
    R_final, Psi_final = osc.order_parameter()
    print(f"  Beigas: R = {R_final:.4f}")
    print(f"  Q_joy  = {q_joy(osc):.4f}")

    weaver = AudioWeaver(duration=12.0)
    audio = weaver.synthesize_from_oscillator(osc)

    path1 = weaver.save_wav(audio, "phaseflow_resonance.wav")
    print(f"  Audio: {path1}")
    print()

    # ── Test 2: Mantra "AUM" → audio ──────────────────────
    print("▸ Test 2: Mantra 'AUM AUM AUM' → audio")
    print("-" * 40)

    audio2, path2 = weave_mantra(
        text="AUM AUM AUM",
        duration=15.0,
        output_path="mantra_aum.wav"
    )
    print()

    # ── Test 3: Čakru drone ───────────────────────────────
    print("▸ Test 3: Čakru drone (meditatīvs akords)")
    print("-" * 40)

    path3 = create_chakra_drone(duration=20.0, output_path="chakra_meditation.wav")
    print()

    # ── Info par failiem ──────────────────────────────────
    print("=" * 60)
    print("  Izveidotie audio faili:")
    for p in [path1, path2, path3]:
        size_kb = os.path.getsize(p) / 1024
        print(f"    → {p.name} ({size_kb:.1f} KB)")
    print()
    print("  Atskaņo ar: aplay <fails.wav> vai mpv <fails.wav>")
    print("=" * 60)
