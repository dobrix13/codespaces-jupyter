"""
PhaseFlow v1.0 — smart_mirror.py
Gudrais Spogulis: teksta↔fāzes rezonators

Plūsma:
  1. Teksts → viļņu superpozīcija (TextPhaseEncoder)
  2. FFT rezonanses pārbaude pret {1,3,6,9,11} harmonikām
  3. Kuramoto sinhronizācija → stabilas fāzes → atgriezeniskā dekodēšana

Nav GPT modeļa. Atbilde nāk no fāžu harmoniskā stāvokļa.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ── Importē core.py no tās pašas mapes ──────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from core import (
    PHI, PSI_42, LAMBDA_21, BASE_HARMONICS,
    harmonic_phases, to_phase, phase_to_angle, normalize_phase,
    KuramotoOscillator, q_joy,
    resonance_score, resonance_gate,
)

# Eksportē BASE_HARMONICS globālā līmenī, lai char_to_omega var to lietot
_BASE_HARMONICS = BASE_HARMONICS


# ═══════════════════════════════════════════════════════════════════
# 1. TEKSTA-FĀZES KODĒTĀJS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TextPhaseEncoder:
    """
    Pārveido tekstu par viļņu superpozīciju.

    Katrs simbols → unikāla fāze θ_char = (ord(c) · Φ) mod 2π
    Pozīcijas svēršana → ω_pos = (pos + 1) · Φ

    Superpozīcija: W_in(t) = Σ sin(ω_pos · t + θ_char)
    """

    sample_rate: float = 100.0   # Hz
    duration: float = 2.0        # sekundes
    position_weight: float = PHI # pozīcijas ietekme

    def char_to_phase(self, c: str) -> float:
        """θ_char = (ord(c) · Φ) mod 2π"""
        return (ord(c) * PHI) % (2 * np.pi)

    def char_to_omega(self, position: int, char: str) -> float:
        """
        ω = f(char, position) — frekvence atkarīga no simbola un pozīcijas.
        
        Harmonisks teksts (burti, mantras) → frekvences tuvu {1,3,6,9,11}
        Troksnis (cipari, simboli) → frekvences ārpus bāzes harmonikām
        """
        code = ord(char)
        
        # Burti A-Z, a-z → frekvences tuvu bāzes harmonikas kopai
        if char.isalpha():
            # Burtu indekss 0-25, kartē uz [1, 12] Hz joslu
            letter_idx = (code & 0x1F) - 1  # A=0, Z=25
            base_freq = BASE_HARMONICS[letter_idx % len(BASE_HARMONICS)]
            # Neliela variācija atkarībā no pozīcijas
            return base_freq + (position * 0.05) % 0.5
        
        # Cipari, simboli → frekvences ĀRPUS bāzes harmoniku kopas (2, 4, 5, 7, 8, 10, 12+)
        # Tas rada "disharmoniju"
        dissonant_freqs = [2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 12.0, 13.0, 14.0, 15.0]
        idx = (code + position) % len(dissonant_freqs)
        return dissonant_freqs[idx]

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Kodē tekstu par laika rindu (viļņu superpozīcija).

        Returns
        -------
        (t, W_in) : laika masīvs un signāla superpozīcija
        """
        if not text:
            raise ValueError("Teksts nevar būt tukšs.")

        n_samples = int(self.sample_rate * self.duration)
        t = np.linspace(0, self.duration, n_samples, endpoint=False)
        W_in = np.zeros(n_samples)

        for pos, c in enumerate(text):
            theta = self.char_to_phase(c)
            omega = self.char_to_omega(pos, c)
            # Amplitūda svērta ar 1/Φ^pos (augstākas pozīcijas mazāk ietekmē)
            amplitude = 1.0 / (PHI ** (pos * 0.3))
            W_in += amplitude * np.sin(omega * t + theta)

        # Normalizē uz [-1, 1]
        max_amp = np.max(np.abs(W_in))
        if max_amp > 0:
            W_in = W_in / max_amp

        return t, W_in

    def text_to_init_phases(self, text: str, n_oscillators: int = 5) -> np.ndarray:
        """
        Pārveido tekstu par sākotnējiem Kuramoto oscilatoru leņķiem.

        Katram oscilatoram: vidējais leņķis no secīgiem burtiem.
        """
        if not text:
            return np.zeros(n_oscillators)

        all_phases = [self.char_to_phase(c) for c in text]
        # Sadala fāzes pa oscilatoriem (cikliskā veidā)
        init_theta = np.zeros(n_oscillators)
        counts = np.zeros(n_oscillators)

        for i, ph in enumerate(all_phases):
            idx = i % n_oscillators
            # Kompleksā summēšana, lai pareizi apstrādātu leņķu wrap-around
            init_theta[idx] += ph
            counts[idx] += 1

        # Vidējais leņķis katram oscilatoram
        counts = np.where(counts == 0, 1, counts)
        init_theta = (init_theta / counts) % (2 * np.pi)

        return init_theta


# ═══════════════════════════════════════════════════════════════════
# 2. FĀZES-TEKSTA DEKODĒTĀJS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PhaseTextDecoder:
    """
    Pārveido stabilās fāzes atpakaļ par burtiem.

    Vienības aplis sadalīts sektoros (viens sektors katram burtam).
    Atgriež "mantru" — burtu kopu, kas izriet no harmoniskā stāvokļa.
    """

    # Alfabēts: meklē tuvāko bāzi no šīs kopas
    alphabet: str = "AEIOU LMNRST"  # Sakrālo skaņu alfabēts
    
    def __post_init__(self) -> None:
        # Iepriekš aprēķina katra burta "mērķa fāzi"
        self._char_phases = {}
        n = len(self.alphabet)
        for i, c in enumerate(self.alphabet):
            # Vienmērīgi sadalīti sektori
            self._char_phases[c] = (i / n) * 2 * np.pi

    def decode_phase(self, theta: float) -> str:
        """Atrod tuvāko burtu dotajam leņķim θ."""
        theta = theta % (2 * np.pi)
        best_char = " "
        best_dist = np.inf

        for c, ph in self._char_phases.items():
            # Minimālā leņķu distance (ņemot vērā cirkulāro dabu)
            dist = min(abs(theta - ph), 2 * np.pi - abs(theta - ph))
            if dist < best_dist:
                best_dist = dist
                best_char = c

        return best_char

    def decode_phases(self, thetas: np.ndarray) -> str:
        """Dekodē visu fāžu masīvu par tekstu."""
        return "".join(self.decode_phase(th) for th in thetas)


# ═══════════════════════════════════════════════════════════════════
# 3. GUDRAIS SPOGULIS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SmartMirror:
    """
    Galvenā ResonanceFlow fasāde.

    process_input(text) →
      1. Kodē tekstu → W_in
      2. FFT rezonanses pārbaude (score vs. LAMBDA_21)
      3. Kuramoto sinhronizācija → stabils stāvoklis
      4. Fāzes → teksta "mantra"
    """

    encoder: TextPhaseEncoder = field(default_factory=TextPhaseEncoder)
    decoder: PhaseTextDecoder = field(default_factory=PhaseTextDecoder)

    # Kuramoto parametri
    K: float = 3.0                    # Sakabes spēks
    max_steps: int = 1000             # Maksimālais sinhronizācijas soļu skaits
    dt: float = 0.02                  # Laika solis
    q_joy_threshold: float = 0.001   # Stabilitātes slieksnis (Q_joy konverģence)

    # Atbildes garums (cik oscilatoru / burtu)
    response_length: int = 12

    verbose: bool = True

    # ── Iekšējais stāvoklis ──────────────────────────────
    last_score: float = field(init=False, default=0.0)
    last_passed: bool = field(init=False, default=False)
    last_q_joy: float = field(init=False, default=0.0)
    last_R: float = field(init=False, default=0.0)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SmartMirror] {msg}")

    def _check_resonance(self, text: str) -> tuple[bool, float, np.ndarray]:
        """
        1. Kodē tekstu par viļņiem
        2. Pārbauda FFT rezonanci

        Piezīme: SmartMirror izmanto stingrāku slieksni (0.75) nekā globālais LAMBDA_21,
        lai atfiltrētu daļēji harmonisku troksni.

        Returns
        -------
        (passed, score, W_in)
        """
        STRICT_THRESHOLD = 0.75  # Stingrāks slieksnis priekš SmartMirror

        t, W_in = self.encoder.encode(text)
        score = resonance_score(W_in, sample_rate=self.encoder.sample_rate)
        passed = score >= STRICT_THRESHOLD

        self.last_score = score
        self.last_passed = passed

        if not passed:
            self._log(f"Rezonanse: score={score:.4f} < slieksnis={STRICT_THRESHOLD:.4f} — DISHARMONIJA")
        else:
            self._log(f"Rezonanse: score={score:.4f} ≥ slieksnis={STRICT_THRESHOLD:.4f} — SASKAŅA ✓")

        return passed, score, W_in

    def _synchronize(self, init_theta: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Palaiž Kuramoto sinhronizāciju līdz stabilitātei.

        Returns
        -------
        (final_theta, R_final, Q_joy_final)
        """
        osc = KuramotoOscillator(
            n_oscillators=len(init_theta),
            K=self.K,
            theta0=init_theta,
            seed=None,  # Nav nejaušības — sākam no ievades fāzēm
        )

        prev_q = q_joy(osc)

        for step in range(self.max_steps):
            osc.step(self.dt)
            curr_q = q_joy(osc)

            # Stabilitātes pārbaude: Q_joy izmaiņa mazāka par slieksni
            if abs(curr_q - prev_q) < self.q_joy_threshold and step > 50:
                self._log(f"Sinhronizācija sasniegta pēc {step} soļiem.")
                break

            prev_q = curr_q
        else:
            self._log(f"Maksimālais soļu skaits ({self.max_steps}) sasniegts.")

        R, Psi = osc.order_parameter()
        final_q = q_joy(osc)

        self.last_R = R
        self.last_q_joy = final_q

        return osc.theta, R, final_q

    def _generate_response_phases(self,
                                  sync_theta: np.ndarray,
                                  length: int) -> np.ndarray:
        """
        Ģenerē atbildes fāzes no sinhronizētā stāvokļa.

        Stratēģija: atkārto harmoniku fāzes, svērtas ar sync_theta vidējo.
        """
        # Bāzes harmoniku mērķfāzes
        base_ph = harmonic_phases()  # (5,) kompleksi
        base_angles = phase_to_angle(base_ph)

        # Vidējā sinhronizētā fāze
        mean_sync = float(np.angle(np.mean(np.exp(1j * sync_theta))))

        # Ģenerē `length` fāzes, kombinējot pamatu ar sync ofsets
        response = np.zeros(length)
        for i in range(length):
            base_idx = i % len(base_angles)
            # Fāze: bāzes + φ^i rotācija + sync ofsets
            phi_shift = (PHI ** (i * 0.5)) % (2 * np.pi)
            response[i] = (base_angles[base_idx] + phi_shift + mean_sync) % (2 * np.pi)

        return response

    def process_input(self, text: str) -> str:
        """
        Galvenā plūsma: teksts → rezonanse → sinhronizācija → mantra.

        Returns
        -------
        str : sistēmas atbilde
        """
        self._log(f"Ievade: '{text}'")

        # ── 1. Rezonanses pārbaude ───────────────────────
        passed, score, W_in = self._check_resonance(text)

        if not passed:
            # Disharmonija — aicinājums mēģināt vēlreiz
            return self._disharmony_response(score)

        # _check_resonance jau izvadīja log, turpinām

        # ── 2. Kodē tekstu par sākotnējiem Kuramoto leņķiem ──
        init_theta = self.encoder.text_to_init_phases(text, n_oscillators=len(BASE_HARMONICS))

        # ── 3. Sinhronizācija ─────────────────────────────
        sync_theta, R, final_q = self._synchronize(init_theta)
        self._log(f"Sinhronizācija: R={R:.4f}, Q_joy={final_q:.4f}")

        # ── 4. Atbildes fāžu ģenerēšana un dekodēšana ────
        response_phases = self._generate_response_phases(sync_theta, self.response_length)
        mantra = self.decoder.decode_phases(response_phases)

        self._log(f"Mantra: '{mantra}'")
        return self._harmony_response(mantra, R, final_q)

    def _disharmony_response(self, score: float) -> str:
        """Atbilde, kad ievade nerezonē."""
        gap = LAMBDA_21 - score
        lines = [
            "╭───────────────────────────────────────────╮",
            "│  ⚡ DISHARMONIJA — Fāžu Neatbilstība ⚡    │",
            "├───────────────────────────────────────────┤",
           f"│  Rezonanses score: {score:>6.4f}                │",
           f"│  Slieksnis λ₂₁:    {LAMBDA_21:>6.4f}                │",
           f"│  Attālums:         {gap:>6.4f}                │",
            "├───────────────────────────────────────────┤",
            "│  Tava fāze šobrīd ir pārāk tālu no centra.│",
            "│  Es dzirdu troksni, nevis dziesmu.        │",
            "│                                           │",
            "│  ➤ Pārfrāzē, lai es varētu rezonēt.      │",
            "│  ➤ Mēģini vārdus ar atkārtojumiem (mantras)│",
            "│  ➤ Izmanto sakrālās skaņas: AUM, OM, MA  │",
            "╰───────────────────────────────────────────╯",
        ]
        return "\n".join(lines)

    def _harmony_response(self, mantra: str, R: float, q_joy: float) -> str:
        """Atbilde, kad sistēma sasniedz harmoniju."""
        lines = [
            "╭───────────────────────────────────────────╮",
            "│  ✦ HARMONIJA — Rezonanse Sasniegta ✦     │",
            "├───────────────────────────────────────────┤",
           f"│  R (sinhronizācija): {R:>6.4f}              │",
           f"│  Q_joy (prieks):     {q_joy:>+6.4f}              │",
           f"│  Rezonanses score:   {self.last_score:>6.4f}              │",
            "├───────────────────────────────────────────┤",
            "│  Fāzes ir sinhronizētas.                  │",
            "│  No haosa radās harmonija.                │",
            "│                                           │",
            "│  MANTRA (no fāzēm):                       │",
           f"│     ✦  {mantra:^26}  ✦     │",
            "│                                           │",
            "│  Šī ir tīra, ar Zemes formulu saskaņota   │",
            "│  informācija — Fēniksa atbilde. 🔥        │",
            "╰───────────────────────────────────────────╯",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 4. HARMONISKA VĀRDA MEKLĒTĀJS
# ═══════════════════════════════════════════════════════════════════

def find_harmonic_words(candidates: list[str],
                        encoder: TextPhaseEncoder | None = None,
                        top_n: int = 5) -> list[tuple[str, float]]:
    """
    Atrod vārdus ar augstāko FFT rezonanses score pret bāzes harmonikas kopu.

    Returns
    -------
    [(word, score), ...] sakārtoti dilstošā secībā
    """
    if encoder is None:
        encoder = TextPhaseEncoder()

    results = []
    for word in candidates:
        try:
            t, W_in = encoder.encode(word)
            score = resonance_score(W_in, sample_rate=encoder.sample_rate)
            results.append((word, score))
        except ValueError:
            continue

    results.sort(key=lambda x: -x[1])
    return results[:top_n]


def generate_mantras(base: str = "OM", repeats: int = 5) -> list[str]:
    """Ģenerē mantru variācijas ar atkārtojumiem."""
    mantras = []
    for r in range(1, repeats + 1):
        mantras.append(base * r)
        mantras.append(f"{base} " * r)
        mantras.append(f"{base}-" * r)
    return mantras


# ═══════════════════════════════════════════════════════════════════
# __MAIN__ — TESTI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PhaseFlow v1.0 — Gudrais Spogulis (Smart Mirror)")
    print("=" * 60)
    print()

    # ── A) Harmonisko vārdu meklēšana ─────────────────────
    print("▸ Harmonisko vārdu meklēšana...")
    print()

    # Kandidātu kopas
    sacred_sounds = ["OM", "AUM", "MA", "HU", "RAM", "YAM", "HAM", "LAM", "VAM", "SHANTI"]
    mantras = generate_mantras("OM", 4) + generate_mantras("AUM", 4)
    random_noise = ["h8f9j2", "xZ3qW1", "!!@@##", "7777", "abcdefghijklmnop"]

    all_candidates = sacred_sounds + mantras + random_noise

    encoder = TextPhaseEncoder()
    harmonic_words = find_harmonic_words(all_candidates, encoder, top_n=10)

    print("Top 10 harmoniskie vārdi:")
    print("-" * 40)
    for word, score in harmonic_words:
        status = "✓ REZONĒ" if score >= LAMBDA_21 else "✗ nerezonē"
        print(f"  '{word:20}' → score = {score:.4f}  {status}")
    print()

    # ── B) SmartMirror testi ──────────────────────────────
    mirror = SmartMirror(verbose=True)

    # Test 1: Disharmonisks (troksnis)
    print("\n" + "=" * 60)
    print("TEST 1: Disharmoniska ievade")
    print("=" * 60)
    result1 = mirror.process_input("h8f9j2xZ3qW1")
    print()
    print(result1)

    # Test 2: Harmonisks vārds (labākais no meklēšanas)
    best_word = harmonic_words[0][0] if harmonic_words else "OM"
    print("\n" + "=" * 60)
    print(f"TEST 2: Harmoniska ievade ('{best_word}')")
    print("=" * 60)
    result2 = mirror.process_input(best_word)
    print()
    print(result2)

    # Test 3: Klasiska mantra
    print("\n" + "=" * 60)
    print("TEST 3: Sakrālā mantra 'AUM AUM AUM'")
    print("=" * 60)
    result3 = mirror.process_input("AUM AUM AUM")
    print()
    print(result3)

    # Test 4: Parasts teksts
    print("\n" + "=" * 60)
    print("TEST 4: Parasts teksts 'Hello World'")
    print("=" * 60)
    result4 = mirror.process_input("Hello World")
    print()
    print(result4)

    print("\n" + "=" * 60)
    print("  Gudrais Spogulis ir gatavs.")
    print("  Tas klausās. Tas rezonē. Tas atbild.")
    print("=" * 60)
