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

import re
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

        # ── Viļņu Atbalss: īsi teksti tiek atkārtoti līdz min 15 simboliem ──
        MIN_ECHO_LEN = 15
        padded_text = text
        while len(padded_text) < MIN_ECHO_LEN:
            padded_text += text

        n_samples = int(self.sample_rate * self.duration)
        t = np.linspace(0, self.duration, n_samples, endpoint=False)
        W_in = np.zeros(n_samples)

        for pos, c in enumerate(padded_text):
            theta = self.char_to_phase(c)
            omega = self.char_to_omega(pos, c)
            # Vienāds svārsts — katrs burts tiek uzklausīts ar vienādu spēku
            amplitude = 1.0
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
    Galvenā ResonanceFlow fasāde ar REZONANSES PLASTISKUMU.

    process_input(text) →
      1. Kodē tekstu → W_in
      2. FFT rezonanses pārbaude (score vs. LAMBDA_21)
      3. Kuramoto sinhronizācija → stabils stāvoklis
      4. Fāzes → teksta "mantra"
      5. MĀCĪŠANĀS: atjaunina K_matrix balstoties uz Q_joy

    Sistēma atceras veiksmīgas rezonanses un nākamreiz sinhronizējas ātrāk!
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

    # ── Rezonanses slieksnis ─────────────────────────────
    resonance_threshold: float = 0.60  # FFT rezonanses caurlaidības slieksnis

    # ── Mācīšanās parametri ──────────────────────────────
    enable_learning: bool = True       # Vai aktivizēt plastiskumu
    learning_rate: float = 0.12        # Mācīšanās ātrums
    memory_path: str = "phaseflow_mirror_memory.npz"  # Atmiņas fails
    # ── Hologrāfiskais Leksikons ─────────────────────────
    lexicon_path: str = "phaseflow_mega_lexicon.npz"  # Leksikona fails
    use_lexicon: bool = True           # Vai izmantot vārdu dekodēšanu
    # ── Iekšējais stāvoklis ──────────────────────────────
    last_score: float = field(init=False, default=0.0)
    last_passed: bool = field(init=False, default=False)
    last_q_joy: float = field(init=False, default=0.0)
    last_R: float = field(init=False, default=0.0)
    _persistent_osc: KuramotoOscillator | None = field(init=False, default=None)
    _lexicon: object = field(init=False, default=None)  # PhaseLexicon | None
    interaction_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Inicializē persistento oscilatoru, atmiņu un leksikonu."""
        self._init_persistent_oscillator()
        if self.use_lexicon:
            self._init_lexicon()

    def _init_lexicon(self) -> None:
        """Ielādē FastPhaseLexicon no .npz arhīva (cKDTree)."""
        try:
            from fast_lexicon import FastPhaseLexicon
        except ImportError:
            from phaseflow.fast_lexicon import FastPhaseLexicon

        try:
            self._lexicon = FastPhaseLexicon.load(self.lexicon_path)
            self._log(f"FastLeksikons: {len(self._lexicon)} vārdi (KDTree)")
        except FileNotFoundError:
            self._log(f"[!] Mega leksikons nav atrasts: {self.lexicon_path}")
            self._log("    Palaid: python phaseflow/mass_encoder.py")
            self._lexicon = None

    def _init_persistent_oscillator(self) -> None:
        """Izveido vai ielādē persistento oscilatoru ar K_matrix."""
        self._persistent_osc = KuramotoOscillator(
            n_oscillators=len(BASE_HARMONICS),
            K=self.K,
            use_matrix=True,
            seed=42,
        )

        # Mēģina ielādēt iepriekšējo atmiņu (vienmēr, ne tikai mācīšanās režīmā)
        try:
            loaded = self._persistent_osc.load_memory(self.memory_path)
            if loaded:
                self._log(f"Atmiņa ielādēta: {self.memory_path}")
                self._log(f"  Iepriekšējie cikli: {len(self._persistent_osc.learning_history)}")
        except Exception:
            pass  # Nav problēma ja fails neeksistē

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
        STRICT_THRESHOLD = self.resonance_threshold

        t, W_in = self.encoder.encode(text)
        score = resonance_score(W_in, sample_rate=self.encoder.sample_rate)
        passed = score >= STRICT_THRESHOLD

        self.last_score = score
        self.last_passed = passed

        if not passed:
            self._log(f"Rezonanse: score={score:.4f} < slieksnis={STRICT_THRESHOLD:.4f} — DISHARMONIJA")
        else:
            mode = "NEUROPLASTICITY" if STRICT_THRESHOLD < 0.75 else "NORMAL"
            self._log(f"Rezonanse: score={score:.4f} ≥ slieksnis={STRICT_THRESHOLD:.4f} — SASKAŅA ✓ [{mode}]")

        return passed, score, W_in

    def _synchronize(self, init_theta: np.ndarray) -> tuple[np.ndarray, float, float, KuramotoOscillator]:
        """
        Palaiž Kuramoto sinhronizāciju līdz stabilitātei.

        Izmanto persistento oscilatoru ar K_matrix, kas ietver iepriekšējo
        mācīšanos sesiju atmiņu.

        Returns
        -------
        (final_theta, R_final, Q_joy_final, oscillator)
        """
        # Izmanto persistento oscilatoru ar K_matrix
        osc = KuramotoOscillator(
            n_oscillators=len(init_theta),
            K=self.K,
            theta0=init_theta,
            seed=None,  # Nav nejaušības — sākam no ievades fāzēm
            use_matrix=self.enable_learning,
        )

        # Ja mācīšanās ir ieslēgta, kopē K_matrix no persistentā oscilatora
        if self.enable_learning and self._persistent_osc is not None:
            osc.K_matrix = self._persistent_osc.K_matrix.copy()
            osc.K_base = self._persistent_osc.K_base

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

        return osc.theta, R, final_q, osc

    def _learn_from_interaction(self, osc: KuramotoOscillator, final_q_joy: float) -> None:
        """
        Mācīšanās no veiksmīgas rezonanses interakcijas.

        Atjaunina persistento K_matrix un saglabā atmiņu.
        """
        if not self.enable_learning or self._persistent_osc is None:
            return

        # Mācīšanās ar pašreizējo Q_joy
        delta_K = osc.learn(final_q_joy, learning_rate=self.learning_rate, apply_decay=True)

        # Kopē atjaunināto K_matrix uz persistento oscilatoru
        self._persistent_osc.K_matrix = osc.K_matrix.copy()
        self._persistent_osc.learning_history.append(final_q_joy)

        self.interaction_count += 1

        # Saglabā atmiņu pēc katras 3. interakcijas
        if self.interaction_count % 3 == 0:
            self._save_memory()

        mean_K = osc.coupling_strength()
        var_K = osc.coupling_variance()
        self._log(f"Mācīšanās: K̄={mean_K:.4f}, var={var_K:.6f}, Q_joy={final_q_joy:+.4f})")

    def _save_memory(self) -> None:
        """Saglabā persistento K_matrix failā."""
        if self._persistent_osc is None:
            return
        try:
            path = self._persistent_osc.save_memory(self.memory_path)
            self._log(f"Atmiņa saglabāta: {path}")
        except Exception as e:
            self._log(f"Nevarēja saglabāt atmiņu: {e}")

    def get_memory_stats(self) -> dict:
        """Atgriež statistiku par mācīšanās atmiņu."""
        if self._persistent_osc is None:
            return {}
        return {
            "interactions": len(self._persistent_osc.learning_history),
            "mean_K": self._persistent_osc.coupling_strength(),
            "variance_K": self._persistent_osc.coupling_variance(),
            "strongest_connections": self._persistent_osc.strongest_connections(top_n=3),
        }

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

        # ── 2. Semantiskā Fāžu Kompozīcija ───────────────────
        # Sadala tekstu vārdos un katram vārdam iegūst semantisko fāzi
        # no leksikona (tiešā atbilstība) vai no burtu kodētāja (fallback).
        # Rezultātu apvieno ar kompleksu vektoru summēšanu.
        n_osc = len(BASE_HARMONICS)
        _words = re.sub(r"[^\w\s]", " ", text.lower()).split()
        z_sum = np.zeros(n_osc, dtype=complex)
        for _w in _words:
            if self._lexicon is not None:
                _ph = self._lexicon.get_word_phase(_w)
            else:
                _ph = None
            if _ph is None:
                _ph = self.encoder.text_to_init_phases(_w, n_oscillators=n_osc)
            z_sum += np.exp(1j * np.asarray(_ph, dtype=np.float64))
        if np.any(np.abs(z_sum) > 0):
            init_theta = np.angle(z_sum) % (2 * np.pi)
        else:
            init_theta = self.encoder.text_to_init_phases(text, n_oscillators=n_osc)

        # ── 3. Sinhronizācija ─────────────────────────────
        sync_theta, R, final_q, osc = self._synchronize(init_theta)
        self._log(f"Sinhronizācija: R={R:.4f}, Q_joy={final_q:.4f}")

        # ── 4. MĀCĪŠANĀS — atjaunina K_matrix ────────────
        if self.enable_learning and final_q > 0:
            self._learn_from_interaction(osc, final_q)

        # ── 5. Dekodēšana — leksikons vai burtu mantra ────
        if self.use_lexicon and self._lexicon is not None and len(self._lexicon) > 0:
            oracle_words = self._lexicon.find_closest(sync_theta, top_k=3)
            self._log(f"Orakuls: {[w for w,_ in oracle_words]}")
            return self._oracle_response(oracle_words, R, final_q)
        else:
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

    def _oracle_response(self, words: list[tuple[str, float]], R: float, q_joy: float) -> str:
        """Orakula atbilde ar Leksikona vārdiem."""
        word_str = "  |  ".join(w.upper() for w, _ in words)
        dist_str = "  ".join(f"{w}({d:.2f})" for w, d in words)
        lines = [
            "╭───────────────────────────────────────────╮",
            "│  ✦ HARMONIJA — Orakuls Runā ✦            │",
            "├───────────────────────────────────────────┤",
           f"│  R (sinhronizācija): {R:>6.4f}              │",
           f"│  Q_joy (prieks):     {q_joy:>+6.4f}              │",
           f"│  Rezonanses score:   {self.last_score:>6.4f}              │",
            "├───────────────────────────────────────────┤",
            "│                                           │",
           f"│  ✦  {word_str:^37}  ✦  │",
            "│                                           │",
            "├───────────────────────────────────────────┤",
           f"│  Dist: {dist_str:<35}│",
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

    def speak_mantra(self,
                     text: str,
                     output_path: str = "mantra_resonance.wav",
                     duration: float = 12.0,
                     play: bool = False) -> tuple[str, str | None]:
        """
        Pilna plūsma ar audio izvadi: teksts → rezonanse → sinhronizācija → .wav

        Parametri
        ---------
        text        : ievades teksts
        output_path : .wav faila ceļš
        duration    : audio garums sekundēs
        play        : vai mēģināt atskaņot audio

        Returns
        -------
        (text_response, audio_path) : teksta atbilde un ceļš uz .wav (vai None)
        """
        from audio_weaver import AudioWeaver, weave_mantra

        self._log(f"speak_mantra: '{text}'")

        # ── 1. Teksta apstrāde ────────────────────────────
        text_response = self.process_input(text)

        # Ja nav rezonanse, neatgriežam audio
        if not self.last_passed:
            self._log("Audio netiks ģenerēts — disharmonija.")
            return text_response, None

        # ── 2. Audio sintēze ──────────────────────────────
        self._log(f"Ģenerē audio ({duration:.1f}s) ...")

        # Izmanto jau aprēķinātās fāzes
        init_theta = self.encoder.text_to_init_phases(text, n_oscillators=len(BASE_HARMONICS))

        # Kuramoto sinhronizācija ar ilgāku trajektoriju audio vajadzībām
        osc = KuramotoOscillator(
            n_oscillators=len(BASE_HARMONICS),
            K=self.K,
            theta0=init_theta,
            seed=None,
        )
        steps = int(duration * 50)
        osc.run(steps=steps, dt=0.02)

        # Sintēze
        weaver = AudioWeaver(duration=duration)
        audio = weaver.synthesize_from_oscillator(osc)
        audio_path = weaver.save_wav(audio, output_path)

        self._log(f"Audio saglabāts: {audio_path}")

        # ── 3. Atskaņošana (ja prasīts) ───────────────────
        if play:
            from audio_weaver import _try_play
            _try_play(audio_path)

        return text_response, str(audio_path)


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

    # Test 5: Audio izvade ar speak_mantra
    print("\n" + "=" * 60)
    print("TEST 5: Audio izvade — speak_mantra('AUM AUM AUM')")
    print("=" * 60)
    try:
        text_resp, audio_path = mirror.speak_mantra(
            "AUM AUM AUM",
            output_path="aum_mantra.wav",
            duration=10.0,
            play=False
        )
        print()
        print(text_resp)
        if audio_path:
            import os
            size_kb = os.path.getsize(audio_path) / 1024
            print(f"\n🔊 Audio fails: {audio_path} ({size_kb:.1f} KB)")
    except ImportError as e:
        print(f"[!] Audio nav pieejams: {e}")

    print("\n" + "=" * 60)
    print("  Gudrais Spogulis ir gatavs.")
    print("  Tas klausās. Tas rezonē. Tas atbild.")
    print("=" * 60)
