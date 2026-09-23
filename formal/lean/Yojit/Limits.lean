/-
  Formal model of `src/yojit/classify.py`, part 2: `estimate_limits_from_config`.

  FIDELITY NOTE.  Python computes `headroom_gb * 2**30 * 0.25` in IEEE-754 and
  truncates with `int(...)`.  Here the same quantity is built with a single
  truncating integer division (`* 2^30 / 400`), which can differ from the float
  result by a few bytes.  Every theorem in this file is insensitive to that
  difference: only `>= 0`, monotonicity in `ram`, and anti-monotonicity in
  `weight` are used.  See `formal/README.md`.

  PROOF NOTE.  `omega`'s treatment of `min`/`max` is incomplete -- it cannot
  even prove `min a X <= a`.  All lattice reasoning below therefore goes through
  the explicit `Nat.le_min` / `Nat.min_le_left` / `Nat.le_max_left` family.  The
  mirror constants are likewise opaque to `omega`, so a hypothesis phrased with
  a constant is always re-bound to a numeral before arithmetic runs.
-/
import Yojit.Tier

namespace Yojit

abbrev MIN_CONTEXT : Nat := 4096
abbrev MAX_CONTEXT_HARD_CAP : Nat := 65536
abbrev MAX_OUTPUT_HARD_CAP : Nat := 4096
/-- Retained only so `legacyOutputOf` can state what the pre-fix formula did.
Nothing in `outputOf` refers to it any more. -/
abbrev MIN_OUTPUT : Nat := 1024
abbrev CONTEXT_ROUND_TO : Nat := 4096
abbrev SAFETY_FACTOR_DENOM : Nat := 4
abbrev HEADROOM_FLOOR_UNITS : Nat := 100

/-! ### Headroom -/

/-- `max(ram - weight - RESERVED_OS_GB, 1.0)`, in units of 0.01 GiB. -/
def headroomUnits (w r : Nat) : Nat := max (r - w - RESERVED_OS_UNITS) HEADROOM_FLOOR_UNITS

/-- Headroom in bytes: GiB * 2^30, times the 0.25 safety factor. -/
def headroomBytes (w r : Nat) : Nat := headroomUnits w r * (2 ^ 30) / (SCALE * SAFETY_FACTOR_DENOM)

theorem headroomUnits_mono_ram {w r₁ r₂ : Nat} (h : r₁ ≤ r₂) :
    headroomUnits w r₁ ≤ headroomUnits w r₂ := by
  unfold headroomUnits
  exact Nat.max_le.mpr
    ⟨Nat.le_trans (Nat.sub_le_sub_right (Nat.sub_le_sub_right h w) RESERVED_OS_UNITS)
        (Nat.le_max_left _ _),
      Nat.le_max_right _ _⟩

theorem headroomUnits_anti_weight {w₁ w₂ r : Nat} (h : w₁ ≤ w₂) :
    headroomUnits w₂ r ≤ headroomUnits w₁ r := by
  unfold headroomUnits
  exact Nat.max_le.mpr
    ⟨Nat.le_trans (Nat.sub_le_sub_right (Nat.sub_le_sub_left h r) RESERVED_OS_UNITS)
        (Nat.le_max_left _ _),
      Nat.le_max_right _ _⟩

theorem headroomBytes_mono_ram {w r₁ r₂ : Nat} (h : r₁ ≤ r₂) :
    headroomBytes w r₁ ≤ headroomBytes w r₂ := by
  unfold headroomBytes
  exact Nat.div_le_div_right (Nat.mul_le_mul_right _ (headroomUnits_mono_ram h))

theorem headroomBytes_anti_weight {w₁ w₂ r : Nat} (h : w₁ ≤ w₂) :
    headroomBytes w₂ r ≤ headroomBytes w₁ r := by
  unfold headroomBytes
  exact Nat.div_le_div_right (Nat.mul_le_mul_right _ (headroomUnits_anti_weight h))

/-- `max_ctx_by_mem`: headroom bytes over bytes of KV cache per token.  The
`kv = 0` branch is the source's `kv_bytes_per_token > 0` guard, falling back to
the model's native context. -/
def maxCtxByMem (w r kv native : Nat) : Nat :=
  if kv = 0 then native else headroomBytes w r / kv

/- Core has no `max_le_max_right` / `min_le_min_right`; these two close the gap and
carry all the nested lattice reasoning for `contextRaw` below. -/
theorem max_mono_right {a b c : Nat} (h : a ≤ b) : max c a ≤ max c b :=
  Nat.max_le.mpr ⟨Nat.le_max_left _ _, Nat.le_trans h (Nat.le_max_right _ _)⟩

theorem min_mono_right {a b c : Nat} (h : a ≤ b) : min c a ≤ min c b :=
  Nat.le_min.mpr ⟨Nat.min_le_left _ _, Nat.le_trans (Nat.min_le_right _ _) h⟩

theorem maxCtxByMem_mono_ram {w r₁ r₂ kv native : Nat} (h : r₁ ≤ r₂) :
    maxCtxByMem w r₁ kv native ≤ maxCtxByMem w r₂ kv native := by
  unfold maxCtxByMem
  by_cases hk : kv = 0
  · rw [if_pos hk, if_pos hk]; exact Nat.le_refl _
  · rw [if_neg hk, if_neg hk]
    exact Nat.div_le_div_right (headroomBytes_mono_ram h)

theorem maxCtxByMem_anti_weight {w₁ w₂ r kv native : Nat} (h : w₁ ≤ w₂) :
    maxCtxByMem w₂ r kv native ≤ maxCtxByMem w₁ r kv native := by
  unfold maxCtxByMem
  by_cases hk : kv = 0
  · rw [if_pos hk, if_pos hk]; exact Nat.le_refl _
  · rw [if_neg hk, if_neg hk]
    exact Nat.div_le_div_right (headroomBytes_anti_weight h)

/-! ### The un-rounded context estimate -/

/-- Mirror of `context = min(native, cap, max(MIN_CONTEXT, max_ctx_by_mem))`. -/
def contextRaw (w r kv native : Nat) : Nat :=
  min native (min MAX_CONTEXT_HARD_CAP (max MIN_CONTEXT (maxCtxByMem w r kv native)))

theorem contextRaw_le_native (w r kv native : Nat) : contextRaw w r kv native ≤ native := by
  unfold contextRaw
  exact Nat.min_le_left _ _

theorem contextRaw_le_cap (w r kv native : Nat) : contextRaw w r kv native ≤ MAX_CONTEXT_HARD_CAP := by
  unfold contextRaw
  exact Nat.le_trans (Nat.min_le_right native _) (Nat.min_le_left _ _)

/-- The exact lower bound on the un-rounded estimate. -/
theorem contextRaw_ge_min (w r kv native : Nat) : min native MIN_CONTEXT ≤ contextRaw w r kv native := by
  have hcap : MIN_CONTEXT ≤ MAX_CONTEXT_HARD_CAP := by decide
  have hmax : MIN_CONTEXT ≤ max MIN_CONTEXT (maxCtxByMem w r kv native) := Nat.le_max_left _ _
  have hinner : MIN_CONTEXT ≤ min MAX_CONTEXT_HARD_CAP (max MIN_CONTEXT (maxCtxByMem w r kv native)) :=
    Nat.le_min.mpr ⟨hcap, hmax⟩
  unfold contextRaw
  exact Nat.le_min.mpr
    ⟨Nat.min_le_left _ _, Nat.le_trans (Nat.min_le_right native MIN_CONTEXT) hinner⟩

/-- **`MIN_CONTEXT` is a floor on the memory-derived estimate only.**  With a
native context at or below it, the returned context is exactly that native value. -/
theorem contextRaw_eq_native_of_le_min {w r kv native : Nat} (h : native ≤ MIN_CONTEXT) :
    contextRaw w r kv native = native := by
  have h' : native ≤ (4096 : Nat) := h
  have hcap : native ≤ MAX_CONTEXT_HARD_CAP := Nat.le_trans h' (by decide)
  have hmax : native ≤ max MIN_CONTEXT (maxCtxByMem w r kv native) := Nat.le_trans h' (Nat.le_max_left _ _)
  unfold contextRaw
  exact Nat.le_antisymm (Nat.min_le_left _ _)
    (Nat.le_min.mpr ⟨Nat.le_refl _, Nat.le_min.mpr ⟨hcap, hmax⟩⟩)

theorem contextRaw_mono_ram {w r₁ r₂ kv native : Nat} (h : r₁ ≤ r₂) :
    contextRaw w r₁ kv native ≤ contextRaw w r₂ kv native := by
  have hm := maxCtxByMem_mono_ram (w := w) (kv := kv) (native := native) h
  unfold contextRaw
  exact min_mono_right (min_mono_right (max_mono_right hm))

theorem contextRaw_anti_weight {w₁ w₂ r kv native : Nat} (h : w₁ ≤ w₂) :
    contextRaw w₂ r kv native ≤ contextRaw w₁ r kv native := by
  have hm := maxCtxByMem_anti_weight (r := r) (kv := kv) (native := native) h
  unfold contextRaw
  exact min_mono_right (min_mono_right (max_mono_right hm))

/-! ### Rounding down to a `CONTEXT_ROUND_TO` multiple -/

def roundTo4096 (c : Nat) : Nat :=
  if CONTEXT_ROUND_TO ≤ c then (c / CONTEXT_ROUND_TO) * CONTEXT_ROUND_TO else c

/-- One division step: for `k > 0`, `k <= n` gives `k <= n / k * k`. -/
theorem le_div_mul_self {k n : Nat} (hk : 0 < k) (h : k ≤ n) : k ≤ n / k * k := by
  have h1 : 1 ≤ n / k := (Nat.le_div_iff_mul_le hk).mpr (by simpa using h)
  calc k = 1 * k := (Nat.one_mul k).symm
    _ ≤ n / k * k := Nat.mul_le_mul_right k h1

theorem roundTo4096_le_self (c : Nat) : roundTo4096 c ≤ c := by
  unfold roundTo4096
  by_cases hc : CONTEXT_ROUND_TO ≤ c
  · rw [if_pos hc]; exact Nat.div_mul_le_self _ _
  · rw [if_neg hc]; exact Nat.le_refl _

theorem roundTo4096_mono {a b : Nat} (h : a ≤ b) : roundTo4096 a ≤ roundTo4096 b := by
  unfold roundTo4096
  by_cases ha : CONTEXT_ROUND_TO ≤ a
  · rw [if_pos ha]
    by_cases hb : CONTEXT_ROUND_TO ≤ b
    · rw [if_pos hb]
      exact Nat.mul_le_mul_right _ (Nat.div_le_div_right h)
    · rw [if_neg hb]
      exact (by omega : False).elim
  · rw [if_neg ha]
    by_cases hb : CONTEXT_ROUND_TO ≤ b
    · rw [if_pos hb]
      refine Nat.le_trans (m := CONTEXT_ROUND_TO) (by omega) ?_
      exact le_div_mul_self (by decide) hb
    · rw [if_neg hb]; exact h

/-- Whatever survives rounding is an exact multiple of 4096. -/
theorem roundTo4096_dvd {c : Nat} (h : CONTEXT_ROUND_TO ≤ roundTo4096 c) :
    CONTEXT_ROUND_TO ∣ roundTo4096 c := by
  unfold roundTo4096 at h ⊢
  by_cases hc : CONTEXT_ROUND_TO ≤ c
  · rw [if_pos hc] at h ⊢
    exact ⟨c / CONTEXT_ROUND_TO, Nat.mul_comm _ _⟩
  · rw [if_neg hc] at h
    exact (by omega : False).elim

/-! ### The final context and output -/

def context (w r kv native : Nat) : Nat := roundTo4096 (contextRaw w r kv native)

/-- Output budget for a context window: a strict quarter, capped, never zero.

The `max 1` is the only floor left. The pre-fix formula floored at `MIN_OUTPUT`,
which is `MIN_CONTEXT / 4` -- so for any context below `MIN_CONTEXT` it handed
back a budget larger than a quarter of the window. `legacyOutputOf` below states
that as a theorem, and `output_ratio_holds_below_min_context` states the fix. -/
def outputOf (c : Nat) : Nat := max 1 (min (c / 4) MAX_OUTPUT_HARD_CAP)

theorem context_le_native (w r kv native : Nat) : context w r kv native ≤ native :=
  Nat.le_trans (roundTo4096_le_self _) (contextRaw_le_native w r kv native)

theorem context_le_cap (w r kv native : Nat) : context w r kv native ≤ MAX_CONTEXT_HARD_CAP :=
  Nat.le_trans (roundTo4096_le_self _) (contextRaw_le_cap w r kv native)

/-- The exact two-sided bound: `min(native, 4096) <= context <= min(native, 65536)`. -/
theorem context_bounds (w r kv native : Nat) :
    min native MIN_CONTEXT ≤ context w r kv native ∧ context w r kv native ≤ min native MAX_CONTEXT_HARD_CAP := by
  have hcap : MIN_CONTEXT ≤ MAX_CONTEXT_HARD_CAP := by decide
  constructor
  · show min native MIN_CONTEXT ≤ roundTo4096 (contextRaw w r kv native)
    unfold roundTo4096
    by_cases hc : CONTEXT_ROUND_TO ≤ contextRaw w r kv native
    · rw [if_pos hc]
      refine Nat.le_trans (Nat.min_le_right native MIN_CONTEXT) ?_
      exact le_div_mul_self (by decide) hc
    · rw [if_neg hc]
      exact contextRaw_ge_min w r kv native
  · exact Nat.le_min.mpr ⟨context_le_native w r kv native, context_le_cap w r kv native⟩

theorem context_dvd {w r kv native : Nat} (h : MIN_CONTEXT ≤ context w r kv native) :
    MIN_CONTEXT ∣ context w r kv native :=
  roundTo4096_dvd h

theorem context_mono_ram {w r₁ r₂ kv native : Nat} (h : r₁ ≤ r₂) :
    context w r₁ kv native ≤ context w r₂ kv native := by
  show roundTo4096 (contextRaw w r₁ kv native) ≤ roundTo4096 (contextRaw w r₂ kv native)
  exact roundTo4096_mono (contextRaw_mono_ram h)

theorem context_anti_weight {w₁ w₂ r kv native : Nat} (h : w₁ ≤ w₂) :
    context w₂ r kv native ≤ context w₁ r kv native := by
  show roundTo4096 (contextRaw w₂ r kv native) ≤ roundTo4096 (contextRaw w₁ r kv native)
  exact roundTo4096_mono (contextRaw_anti_weight h)

/-! #### `MIN_CONTEXT` is not a floor on the returned context -/

/-- **Headline.**  A model whose own native context is below `MIN_CONTEXT` gets
exactly that native context back -- so `context >= MIN_CONTEXT` is *not* a
theorem about this function.  The source documents this intent in a comment; the
README's "memory floor" wording does not carry the caveat. -/
theorem min_context_is_not_a_floor {w r kv native : Nat} (h : native < MIN_CONTEXT) :
    context w r kv native = native := by
  have hle : native ≤ MIN_CONTEXT := Nat.le_of_lt h
  have h' : native < (4096 : Nat) := h
  have hraw : contextRaw w r kv native = native := contextRaw_eq_native_of_le_min hle
  show roundTo4096 (contextRaw w r kv native) = native
  rw [hraw]
  unfold roundTo4096
  rw [if_neg (by omega : ¬ ((4096 : Nat) ≤ native))]

/-! #### The output/context ratio -/

theorem outputOf_pos (c : Nat) : 0 < outputOf c := by
  unfold outputOf
  exact Nat.lt_of_lt_of_le (by decide : (0 : Nat) < 1) (Nat.le_max_left _ _)

theorem outputOf_le_cap (c : Nat) : outputOf c ≤ MAX_OUTPUT_HARD_CAP := by
  unfold outputOf
  exact Nat.max_le.mpr ⟨(by decide : (1 : Nat) ≤ 4096), Nat.min_le_right _ _⟩

/-- At or above 4 tokens the floor never binds and output is exactly `c / 4`, capped. -/
theorem outputOf_eq_quarter {c : Nat} (h : 4 ≤ c) :
    outputOf c = min (c / 4) MAX_OUTPUT_HARD_CAP := by
  have h1 : 1 ≤ c / 4 := (Nat.le_div_iff_mul_le (by decide : 0 < 4)).mpr (by omega)
  have hle : 1 ≤ min (c / 4) MAX_OUTPUT_HARD_CAP :=
    Nat.le_min.mpr ⟨h1, (by decide : (1 : Nat) ≤ 4096)⟩
  unfold outputOf
  exact Nat.max_eq_right hle

/-- **The quarter law, unconditionally** (for any context that can hold a token):
output never exceeds a quarter of the window.  The pre-fix version of this
theorem needed the hypothesis `MIN_CONTEXT ≤ c`. -/
theorem outputOf_le_quarter {c : Nat} (h : 4 ≤ c) : 4 * outputOf c ≤ c := by
  rw [outputOf_eq_quarter h]
  have h1 : min (c / 4) MAX_OUTPUT_HARD_CAP ≤ c / 4 := Nat.min_le_left _ _
  have h2 : 4 * (c / 4) ≤ c := by
    rw [Nat.mul_comm 4 (c / 4)]
    exact Nat.div_mul_le_self c 4
  exact Nat.le_trans (Nat.mul_le_mul_left 4 h1) h2

/-- Output never exceeds the window it is a budget for. -/
theorem outputOf_le_self {c : Nat} (h : 0 < c) : outputOf c ≤ c := by
  unfold outputOf
  refine Nat.max_le.mpr ⟨by omega, ?_⟩
  exact Nat.le_trans (Nat.min_le_left _ _) (Nat.div_le_self c 4)

/-! #### The pre-fix formula, kept as a theorem rather than a memory -/

/-- The formula as it was shipped before this change: `max(MIN_OUTPUT, ...)`. -/
def legacyOutputOf (c : Nat) : Nat := max MIN_OUTPUT (min (c / 4) MAX_OUTPUT_HARD_CAP)

theorem legacy_outputOf_eq_min_output_of_lt {c : Nat} (h : c < MIN_CONTEXT) :
    legacyOutputOf c = MIN_OUTPUT := by
  have h' : c < (4096 : Nat) := h
  have h4 : c / 4 < 1024 := (Nat.div_lt_iff_lt_mul (by decide : 0 < 4)).mpr (by omega)
  have hle : min (c / 4) MAX_OUTPUT_HARD_CAP ≤ MIN_OUTPUT :=
    Nat.le_trans (Nat.min_le_left _ _) (Nat.le_of_lt h4)
  unfold legacyOutputOf
  exact Nat.max_eq_left hle

/-- **The bug, stated for all inputs.** Below `MIN_CONTEXT` the old formula
returned exactly 1024, which is more than a quarter of the window -- at native
2048 it is half of it.  This is the theorem `outputOf_le_quarter` makes
unprovable against the shipped function. -/
theorem legacy_output_exceeds_quarter_of_lt {c : Nat} (h : c < MIN_CONTEXT) :
    c < 4 * legacyOutputOf c := by
  rw [legacy_outputOf_eq_min_output_of_lt h]
  have h' : c < (4096 : Nat) := h
  show c < 4 * 1024
  omega

theorem legacy_output_exceeds_quarter_at_2048 : 2048 < 4 * legacyOutputOf 2048 :=
  legacy_output_exceeds_quarter_of_lt (by decide)

/-- **The fix.** A native context below `MIN_CONTEXT` still comes back unchanged
(`MIN_CONTEXT` is a floor on the memory estimate, not on the native ceiling), but
the output budget is now a strict quarter of it -- and the same inputs under the
old formula are still a counterexample, so the two formulas are proved apart. -/
theorem output_ratio_holds_below_min_context {w r kv native : Nat}
    (hn : native < MIN_CONTEXT) (h4 : 4 ≤ native) :
    context w r kv native = native
      ∧ 4 * outputOf (context w r kv native) ≤ context w r kv native
      ∧ context w r kv native < 4 * legacyOutputOf (context w r kv native) := by
  have hctx := min_context_is_not_a_floor (w := w) (r := r) (kv := kv) hn
  refine ⟨hctx, ?_, ?_⟩
  · rw [hctx]; exact outputOf_le_quarter h4
  · rw [hctx]; exact legacy_output_exceeds_quarter_of_lt hn

end Yojit
