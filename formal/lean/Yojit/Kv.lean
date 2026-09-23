/-
  Formal model of `src/yojit/classify.py`, part 3: `default_kv_cache_overrides`.

  FIDELITY NOTE.  Python computes `bytes_per_token_fp16 * (bits / 16)` in
  IEEE-754 and compares `context * bytes_per_token <= headroom_bytes`.  The
  denominator is cleared here instead: `c * f * bits <= h * 16`.  Over exact
  rationals the two are equivalent, and `bits` is 16, 8, or 4, so every product
  involved is exactly representable.  See `formal/README.md`.
-/
import Yojit.Limits

namespace Yojit

/-! `_KV_QUANT_BIT_OPTIONS = (16, 8, 4)` -- highest precision first. -/

/-- Does a KV cache of `bits` bits per element, `c` tokens long, fit in `h` bytes
of headroom at `f` bytes per token unquantized? -/
abbrev fitsKV (h c f bits : Nat) : Prop := c * f * bits ≤ h * 16

/-- Mirror of the selection loop: the highest precision that still fits, else
`4` -- **not** an error, even when `4` does not fit either.  `16` means "no
override needed". -/
def kvBits (h c f : Nat) : Nat :=
  if fitsKV h c f 16 then 16 else if fitsKV h c f 8 then 8 else 4

/-- Where KV quantization should start: the token index at which the *unquantized*
cache would exhaust headroom, clamped to the context length. -/
def kvStart (h c f : Nat) : Nat := if f = 0 then 0 else min (h / f) c

/-- llama.cpp cache-type names; `.get(bits, "q4_0")` in the source. -/
def llamaCacheType (bits : Nat) : String := if bits = 8 then "q8_0" else "q4_0"

/-! ### Lower bit-widths are always easier to fit -/

theorem fitsKV_anti_bits {h c f b₁ b₂ : Nat} (hle : b₂ ≤ b₁) (hb : fitsKV h c f b₁) :
    fitsKV h c f b₂ := by
  show c * f * b₂ ≤ h * 16
  exact Nat.le_trans (Nat.mul_le_mul_left (c * f) hle) hb

theorem fitsKV_mono_headroom {h₁ h₂ c f bits : Nat} (hh : h₁ ≤ h₂) (hb : fitsKV h₁ c f bits) :
    fitsKV h₂ c f bits :=
  Nat.le_trans hb (Nat.mul_le_mul_right 16 hh)

/-! ### What the selection actually returns -/

theorem kvBits_mem (h c f : Nat) : kvBits h c f = 16 ∨ kvBits h c f = 8 ∨ kvBits h c f = 4 := by
  unfold kvBits
  by_cases h16 : fitsKV h c f 16 <;> by_cases h8 : fitsKV h c f 8 <;> simp [h16, h8]

theorem kvBits_16 (h c f : Nat) : kvBits h c f = 16 ↔ fitsKV h c f 16 := by
  unfold kvBits
  by_cases h16 : fitsKV h c f 16 <;> by_cases h8 : fitsKV h c f 8 <;> simp [h16, h8]

theorem kvBits_8 (h c f : Nat) : kvBits h c f = 8 ↔ ¬ fitsKV h c f 16 ∧ fitsKV h c f 8 := by
  unfold kvBits
  by_cases h16 : fitsKV h c f 16 <;> by_cases h8 : fitsKV h c f 8 <;> simp [h16, h8]

theorem kvBits_4 (h c f : Nat) : kvBits h c f = 4 ↔ ¬ fitsKV h c f 16 ∧ ¬ fitsKV h c f 8 := by
  unfold kvBits
  by_cases h16 : fitsKV h c f 16 <;> by_cases h8 : fitsKV h c f 8 <;> simp [h16, h8]

theorem kvBits_le_16 (h c f : Nat) : kvBits h c f ≤ 16 := by
  rcases kvBits_mem h c f with h' | h' | h' <;> rw [h'] <;> omega

theorem kvBits_le_8_of_not_fits {h c f : Nat} (hn : ¬ fitsKV h c f 16) : kvBits h c f ≤ 8 := by
  rcases kvBits_mem h c f with h' | h' | h'
  · exact absurd ((kvBits_16 h c f).mp h') hn
  · rw [h']; omega
  · rw [h']; omega

/-- More headroom never selects a *lower* precision. -/
theorem kvBits_mono_headroom {h₁ h₂ c f : Nat} (hh : h₁ ≤ h₂) :
    kvBits h₁ c f ≤ kvBits h₂ c f := by
  have mono : ∀ {bits : Nat}, fitsKV h₁ c f bits → fitsKV h₂ c f bits :=
    fun hb => fitsKV_mono_headroom hh hb
  rcases kvBits_mem h₂ c f with h16 | h8 | h4
  · rw [h16]; exact kvBits_le_16 h₁ c f
  · rw [h8]
    exact kvBits_le_8_of_not_fits (fun hp => ((kvBits_8 h₂ c f).mp h8).1 (mono hp))
  · rw [h4]
    have hn16 : ¬ fitsKV h₂ c f 16 := ((kvBits_4 h₂ c f).mp h4).1
    have hn8 : ¬ fitsKV h₂ c f 8 := ((kvBits_4 h₂ c f).mp h4).2
    have h4' : kvBits h₁ c f = 4 :=
      (kvBits_4 h₁ c f).mpr ⟨fun hp => hn16 (mono hp), fun hp => hn8 (mono hp)⟩
    calc kvBits h₁ c f = 4 := h4'
      _ ≤ 4 := Nat.le_refl _

/-- **Silent overshoot.**  When even the 4-bit cache does not fit, the answer is
still 4 bits: the function has no branch that reports failure.  A caller
comparing the chosen cache type against real headroom would have to redo this
arithmetic itself, since nothing in the return value signals the shortfall. -/
theorem kvBits_overshoot {h c f : Nat} (hf : ¬ fitsKV h c f 4) : kvBits h c f = 4 :=
  (kvBits_4 h c f).mpr ⟨fun hp => hf (fitsKV_anti_bits (by omega) hp),
    fun hp => hf (fitsKV_anti_bits (by omega) hp)⟩

/-- The `bytes_per_token <= 0` guard in the source: a zero-cost cache always
"fits", so no override is emitted. -/
theorem kvBits_16_of_f_zero (h c : Nat) : kvBits h c 0 = 16 :=
  (kvBits_16 h c 0).mpr (by show c * 0 * 16 ≤ h * 16; omega)

/-! ### `quantized_kv_start` -/

theorem kvStart_le_context (h c f : Nat) : kvStart h c f ≤ c := by
  unfold kvStart
  by_cases hf : f = 0
  · rw [if_pos hf]; omega
  · rw [if_neg hf]; exact Nat.min_le_right _ _

/-- `quantized_kv_start` is only emitted on the 8-bit and 4-bit paths, and both
require fp16 to *not* fit -- in which case `headroom / f < requested context`
already.  Note this is a statement about the *requested* context: the value
actually launched with is `kvPlanContext` below, which can be smaller again, and
that is exactly why `resolve_kv_cache` clamps to it. -/
theorem kvStart_lt_requested_context_of_not_fp16_fits {h c f : Nat}
    (hn : ¬ fitsKV h c f 16) (hf : 0 < f) : kvStart h c f < c := by
  have hlt : h < c * f := by
    have hn' : ¬ (c * f * 16 ≤ h * 16) := hn
    have : h * 16 < c * f * 16 := Nat.lt_of_not_le hn'
    have : 16 * h < 16 * (c * f) := by
      rw [Nat.mul_comm 16 h, Nat.mul_comm 16 (c * f)]; exact this
    exact Nat.lt_of_mul_lt_mul_left this
  have hdiv : h / f < c :=
    Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm f c]; exact hlt)
  unfold kvStart
  rw [if_neg (by omega : ¬ f = 0), Nat.min_eq_left (Nat.le_of_lt hdiv)]
  exact hdiv

/-! ### llama.cpp cache-type mapping -/

theorem llamaCacheType_8 : llamaCacheType 8 = "q8_0" := by decide
theorem llamaCacheType_4 : llamaCacheType 4 = "q4_0" := by decide

/-! ### The project's own test parameters, checked

`tests/test_classify.py::test_default_kv_cache_overrides_falls_back_to_4bit_when_even_8bit_does_not_fit`
uses a 32-layer / 8-KV-head / head_dim-128 model (so `f = 2*32*8*128*2 = 131072`)
at `weight_gb=15`, `ram_gb=24`, `context=16384`.  Headroom is
`max(24 - 15 - 8, 0.1) * 2^30 * 0.25 = 268435456` bytes.  At those exact inputs the
4-bit cache does **not** fit, and the asserted output is still `"4"`. -/

theorem kv_overshoot_at_project_test_parameters :
    ¬ fitsKV 268435456 16384 131072 4 ∧ kvBits 268435456 16384 131072 = 4 := by
  constructor
  · show ¬ (16384 * 131072 * 4 ≤ 268435456 * 16); decide
  · decide

/-- ...and `8` bits does not fit either, so the fallback is forced, not chosen. -/
theorem kv_8bit_does_not_fit_at_project_test_parameters :
    ¬ fitsKV 268435456 16384 131072 16 ∧ ¬ fitsKV 268435456 16384 131072 8 := by
  constructor <;> · show ¬ (_ * _ * _ ≤ 268435456 * 16); decide

/-- What would actually fit at those inputs.  The chosen 4-bit cache exceeds
available headroom; 1-bit-per-element is the largest width that fits. -/
theorem kv_actual_max_fitting_at_project_test_parameters : fitsKV 268435456 16384 131072 1 := by
  show 16384 * 131072 * 1 ≤ 268435456 * 16; decide

/-! ### The fixed contract: `resolve_kv_cache` and `KvPlan`

The functions above model the *selection*.  The bug was never in the selection --
it was that the return value could not express failure, so the overshoot reached
the launcher unnoticed.  The fix is a second layer: a plan that also reports the
context it can actually hold.  This section models that layer. -/

/-- Bytes per token at `bits` bits per element.  Truncating division; the source
computes this in IEEE-754, and `bits` is always 8 or 4 on the paths that use it,
so `f` even makes the two agree exactly.  See `formal/README.md`. -/
def kvBytesPerToken (f bits : Nat) : Nat := f * bits / 16

/-- Tokens the chosen width can hold in `h` bytes of headroom. -/
def kvMaxTokens (h f bits : Nat) : Nat :=
  if kvBytesPerToken f bits = 0 then 0 else h / kvBytesPerToken f bits

/-- Mirror of `KvPlan.context`: the requested context when it fits, otherwise the
largest 4096-multiple that does, never below 1. -/
def kvPlanContext (h c f : Nat) : Nat :=
  if c ≤ kvMaxTokens h f (kvBits h c f) then c
  else max 1 (roundTo4096 (min c (kvMaxTokens h f (kvBits h c f))))

/-- The plan never *raises* the requested context. -/
theorem kvPlanContext_le_requested {h c f : Nat} (hc : 1 ≤ c) : kvPlanContext h c f ≤ c := by
  unfold kvPlanContext
  by_cases hf : c ≤ kvMaxTokens h f (kvBits h c f)
  · rw [if_pos hf]; exact Nat.le_refl _
  · rw [if_neg hf]
    exact Nat.max_le.mpr ⟨hc, Nat.le_trans (roundTo4096_le_self _) (Nat.min_le_left _ _)⟩

/-- When the request fits, the plan is the identity: no shrink, no rounding. -/
theorem kvPlanContext_eq_requested_of_fits {h c f : Nat}
    (hf : c ≤ kvMaxTokens h f (kvBits h c f)) : kvPlanContext h c f = c := by
  unfold kvPlanContext
  rw [if_pos hf]

/-- **The invariant the old contract could not state.**  Whenever at least one
token fits at the chosen width, the context the plan reports fits the headroom
that width has.  Contrast `kvBits_overshoot`: the *bits* are unchanged, so this
is a genuinely new guarantee about the value that gets launched. -/
theorem kvPlanContext_fits_headroom {h c f : Nat}
    (ht : 1 ≤ kvMaxTokens h f (kvBits h c f)) :
    kvPlanContext h c f ≤ kvMaxTokens h f (kvBits h c f) := by
  unfold kvPlanContext
  by_cases hf : c ≤ kvMaxTokens h f (kvBits h c f)
  · rw [if_pos hf]; exact hf
  · rw [if_neg hf]
    exact Nat.max_le.mpr ⟨ht, Nat.le_trans (roundTo4096_le_self _) (Nat.min_le_right _ _)⟩

/-- The guarantee the MLX branch relies on: whatever the clamp produces is a
valid index into the window launched with. -/
theorem kvStartClamped_le_context (h c f : Nat) :
    min (kvStart h c f) (kvPlanContext h c f) ≤ kvPlanContext h c f :=
  Nat.min_le_right _ _

/-! #### The project's own test parameters, before and after -/

/-- The plan at the parameters the project's test asserts on: 16384 requested,
4 bits chosen, headroom 268435456 -- the reported context comes down to 8192. -/
theorem plan_at_project_test_parameters : kvPlanContext 268435456 16384 131072 = 8192 := by
  decide

/-- ...and 8192 tokens at the chosen 4-bit width really do fit: 8192 * 32768 =
268435456, exactly the available headroom.  This is `kvPlanContext_fits_headroom`
instantiated, and it is the property that could not be stated before. -/
theorem plan_fits_at_project_test_parameters :
    8192 * kvBytesPerToken 131072 4 ≤ 268435456 := by decide

/-- The requested 16384 at the same width needs exactly twice the headroom -- the
2x overshoot the old return value shipped without comment. -/
theorem legacy_plan_overshoots_at_project_test_parameters :
    ¬ (16384 * kvBytesPerToken 131072 4 ≤ 268435456) := by decide

end Yojit
