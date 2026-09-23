/-
  Formal model of `src/yojit/classify.py`, part 4: `compute_launch_tuning`.

  FIDELITY NOTE.  `headroom_gb` is a float in Python, and `_headroom_tier_index`
  compares it against the integral ceilings `2, 4, 8, 16`.  For an integral
  ceiling `c`, `h < c ↔ floor h < c`, so the `Nat` model below (with `h` the
  floor of the real headroom) makes exactly the same decision at every input.
  `headroom_gb` is clamped to `>= 1.0`, so the floor is never negative.
-/
import Yojit.Limits

namespace Yojit

/-! ### `_HEADROOM_TIER_GB` and the three lookup tables -/

/-- `_HEADROOM_TIER_GB`, in whole GiB. -/
def HEADROOM_TIER_CEILINGS : List Nat := [2, 4, 8, 16]
def prefillSizes : List Nat := [512, 1024, 2048, 4096, 8192]
def batchSizes : List Nat := [512, 1024, 2048, 2048, 4096]
def ubatchSizes : List Nat := [256, 512, 512, 1024, 2048]

/-- Mirror of `_headroom_tier_index`: index of the first ceiling above `h`, else
the ceiling count. -/
def headroomTierIndex (h : Nat) : Nat :=
  if h < 2 then 0 else if h < 4 then 1 else if h < 8 then 2 else if h < 16 then 3 else 4

/-- The same number, written as "how many ceilings are at or below `h`".  This is
the honest description of the loop, and it is what makes monotonicity provable. -/
def tierIndexCount (h : Nat) : Nat :=
  (if h < 2 then 0 else 1) + (if h < 4 then 0 else 1) + (if h < 8 then 0 else 1)
    + (if h < 16 then 0 else 1)

theorem tierIndex_eq_count (h : Nat) : headroomTierIndex h = tierIndexCount h := by
  have hc : h < 2 ∨ (2 ≤ h ∧ h < 4) ∨ (4 ≤ h ∧ h < 8) ∨ (8 ≤ h ∧ h < 16) ∨ 16 ≤ h := by omega
  unfold headroomTierIndex tierIndexCount
  rcases hc with h1 | ⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩ | h1
  · simp only [if_pos (show h < 2 by omega), if_pos (show h < 4 by omega),
      if_pos (show h < 8 by omega), if_pos (show h < 16 by omega)] <;> omega
  · simp only [if_neg (show ¬ (h < 2) by omega), if_pos (show h < 4 by omega),
      if_pos (show h < 8 by omega), if_pos (show h < 16 by omega)] <;> omega
  · simp only [if_neg (show ¬ (h < 2) by omega), if_neg (show ¬ (h < 4) by omega),
      if_pos (show h < 8 by omega), if_pos (show h < 16 by omega)] <;> omega
  · simp only [if_neg (show ¬ (h < 2) by omega), if_neg (show ¬ (h < 4) by omega),
      if_neg (show ¬ (h < 8) by omega), if_pos (show h < 16 by omega)] <;> omega
  · simp only [if_neg (show ¬ (h < 2) by omega), if_neg (show ¬ (h < 4) by omega),
      if_neg (show ¬ (h < 8) by omega), if_neg (show ¬ (h < 16) by omega)] <;> omega

/-- An upward indicator of a downward-closed predicate is monotone. -/
theorem ite_indicator_mono {P : Nat → Prop} [DecidablePred P]
    (hanti : ∀ a b, a ≤ b → P b → P a) {a b : Nat} (h : a ≤ b) :
    (if P a then 0 else 1) ≤ (if P b then 0 else 1) := by
  by_cases ha : P a
  · rw [if_pos ha]; exact Nat.zero_le _
  · rw [if_neg ha, if_neg (fun hp => ha (hanti a b h hp))]
    exact Nat.le_refl _

theorem tierIndexCount_mono {h₁ h₂ : Nat} (h : h₁ ≤ h₂) : tierIndexCount h₁ ≤ tierIndexCount h₂ := by
  unfold tierIndexCount
  have a2 := ite_indicator_mono (P := fun x => x < 2) (fun a b hab hb => by omega) h
  have a4 := ite_indicator_mono (P := fun x => x < 4) (fun a b hab hb => by omega) h
  have a8 := ite_indicator_mono (P := fun x => x < 8) (fun a b hab hb => by omega) h
  have a16 := ite_indicator_mono (P := fun x => x < 16) (fun a b hab hb => by omega) h
  omega

/-- More headroom never selects a smaller prefill/batch chunking tier. -/
theorem tierIndex_mono {h₁ h₂ : Nat} (h : h₁ ≤ h₂) :
    headroomTierIndex h₁ ≤ headroomTierIndex h₂ := by
  rw [tierIndex_eq_count h₁, tierIndex_eq_count h₂]
  exact tierIndexCount_mono h

/-- The index is always a valid subscript for a 5-entry table. -/
theorem tierIndex_le_four (h : Nat) : headroomTierIndex h ≤ 4 := by
  unfold headroomTierIndex
  split
  · omega
  · split
    · omega
    · split
      · omega
      · split
        · omega
        · omega

/-! ### The tables, and the invariant that keeps them in bounds

`_HEADROOM_TIER_GB` has `n` ceilings and the tables have `n + 1` buckets.  If a
sixth ceiling is ever added, `headroomTierIndex` returns 5 and Python raises
`IndexError` at launch time -- whereas here `tierIndex_lt_table_length` simply
becomes unprovable, and the build breaks instead of the user's first `yojit serve`.
-/

theorem ceilings_length : HEADROOM_TIER_CEILINGS.length = 4 := by decide
theorem prefillSizes_length : prefillSizes.length = 5 := by decide
theorem batchSizes_length : batchSizes.length = 5 := by decide
theorem ubatchSizes_length : ubatchSizes.length = 5 := by decide

/-- The maintenance invariant, stated once: one table entry per bucket. -/
theorem tables_match_bucket_count :
    prefillSizes.length = HEADROOM_TIER_CEILINGS.length + 1
      ∧ batchSizes.length = HEADROOM_TIER_CEILINGS.length + 1
      ∧ ubatchSizes.length = HEADROOM_TIER_CEILINGS.length + 1 := by decide

theorem tierIndex_lt_table_length (h : Nat) : headroomTierIndex h < prefillSizes.length := by
  have := tierIndex_le_four h
  rw [prefillSizes_length]; omega

/-! ### Table lookups -/

def prefillStepSize (i : Nat) : Nat := prefillSizes.getD i 512
def batchSize (i : Nat) : Nat := batchSizes.getD i 512
def ubatchSize (i : Nat) : Nat := ubatchSizes.getD i 256

/-- Each table is non-decreasing in the bucket index. -/
theorem prefillSize_mono {i j : Nat} (h : i ≤ j) (hi : i ≤ 4) (hj : j ≤ 4) :
    prefillStepSize i ≤ prefillStepSize j := by
  have ci : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 := by omega
  have cj : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 := by omega
  unfold prefillStepSize
  rcases ci with rfl | rfl | rfl | rfl | rfl <;> rcases cj with rfl | rfl | rfl | rfl | rfl
  <;> first | decide | (exfalso; omega)

theorem batchSize_mono {i j : Nat} (h : i ≤ j) (hi : i ≤ 4) (hj : j ≤ 4) :
    batchSize i ≤ batchSize j := by
  have ci : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 := by omega
  have cj : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 := by omega
  unfold batchSize
  rcases ci with rfl | rfl | rfl | rfl | rfl <;> rcases cj with rfl | rfl | rfl | rfl | rfl
  <;> first | decide | (exfalso; omega)

theorem ubatchSize_mono {i j : Nat} (h : i ≤ j) (hi : i ≤ 4) (hj : j ≤ 4) :
    ubatchSize i ≤ ubatchSize j := by
  have ci : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 := by omega
  have cj : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 := by omega
  unfold ubatchSize
  rcases ci with rfl | rfl | rfl | rfl | rfl <;> rcases cj with rfl | rfl | rfl | rfl | rfl
  <;> first | decide | (exfalso; omega)

/-- Composite form: the value actually passed to the server is monotone in headroom. -/
theorem prefillStepSize_mono {h₁ h₂ : Nat} (h : h₁ ≤ h₂) :
    prefillStepSize (headroomTierIndex h₁) ≤ prefillStepSize (headroomTierIndex h₂) :=
  prefillSize_mono (tierIndex_mono h) (tierIndex_le_four h₁) (tierIndex_le_four h₂)

/-! ### `prompt_cache_bytes` -/

abbrev PROMPT_CACHE_MIN_UNITS : Nat := 50
abbrev PROMPT_CACHE_MAX_UNITS : Nat := 800

/-- `_PROMPT_CACHE_HEADROOM_FRACTION = 0.4`, as the exact ratio 2/5 it is used as.

Named rather than inlined so the constant-mirror test can tie it to the Python
float. It is the one tuning fraction that is *not* the `SAFETY_FACTOR` the KV
budget uses, which is worth being able to see: two fractions of one headroom
sizing two caches is the pattern that broke the install estimate. -/
abbrev PROMPT_CACHE_FRACTION_NUM : Nat := 2
abbrev PROMPT_CACHE_FRACTION_DENOM : Nat := 5

/-- `min(8.0, max(0.5, headroom_gb * 0.4))`, in units of 0.01 GiB. -/
def promptCacheUnits (h : Nat) : Nat :=
  min PROMPT_CACHE_MAX_UNITS
    (max PROMPT_CACHE_MIN_UNITS
      (h * PROMPT_CACHE_FRACTION_NUM / PROMPT_CACHE_FRACTION_DENOM))

/-- Never below the 0.5 GiB floor, never above the 8 GiB ceiling. -/
theorem promptCacheUnits_bounds (h : Nat) :
    PROMPT_CACHE_MIN_UNITS ≤ promptCacheUnits h ∧ promptCacheUnits h ≤ PROMPT_CACHE_MAX_UNITS := by
  constructor
  · show (50 : Nat) ≤ promptCacheUnits h
    unfold promptCacheUnits
    exact Nat.le_min.mpr ⟨(by omega : (50 : Nat) ≤ 800), Nat.le_max_left _ _⟩
  · unfold promptCacheUnits
    exact Nat.min_le_left _ _

theorem promptCacheUnits_mono {h₁ h₂ : Nat} (h : h₁ ≤ h₂) :
    promptCacheUnits h₁ ≤ promptCacheUnits h₂ := by
  unfold promptCacheUnits
  exact min_mono_right (max_mono_right (Nat.div_le_div_right (Nat.mul_le_mul_right 2 h)))

/-! ### `threads` -/

def threads (cores : Nat) : Nat := max 1 (cores - 1)

theorem threads_ge_one (cores : Nat) : 1 ≤ threads cores := by
  unfold threads; exact Nat.le_max_left _ _

/-- One core is always left for the OS, and single-core machines still get one. -/
theorem threads_le_cores {cores : Nat} (h : 1 ≤ cores) : threads cores ≤ cores := by
  unfold threads
  exact Nat.max_le.mpr ⟨h, by omega⟩

/-! ### Which knobs are *not* spec-derived

The README says "every knob beyond context/output is recomputed fresh from real
RAM headroom and CPU core count on every `serve` call, never a fixed constant".
Three of the returned knobs are literal constants, by design (the source comments
say so), so the README sentence overstates the code. -/

/-- `--gpu-layers` is the literal 999, not a function of the machine. -/
def ngl : Nat := 999
def decodeConcurrency : Nat := 1
def promptConcurrency : Nat := 1

theorem ngl_is_not_spec_derived : ngl = 999 := rfl
theorem decodeConcurrency_is_not_spec_derived : decodeConcurrency = 1 := rfl
theorem promptConcurrency_is_not_spec_derived : promptConcurrency = 1 := rfl

end Yojit
