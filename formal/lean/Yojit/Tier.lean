/-
  Formal model of `src/yojit/classify.py`, part 1: resource tiers and `fits_at_all`.

  UNITS AND FIDELITY
  ------------------
  Sizes are `Nat` in units of 0.01 GiB (`50` = 0.50 GiB, `800` = 8.00 GiB).  Over
  exact rationals, Python's `weight_gb / ram_gb <= 0.35` is equivalent to the
  integer cross-multiplication `100 * w <= 35 * r` used here.  IEEE-754 rounding
  is *not* modelled -- see `formal/README.md` for that documented gap.
-/
namespace Yojit

/-- Resource tier, mirroring `resource_tier`'s three return strings. -/
inductive Tier where
  | low
  | medium
  | high
  deriving DecidableEq, Repr

/-- Ordinal rank, so tier comparisons reduce to `Nat` comparisons. -/
def Tier.rank : Tier → Nat
  | .low => 0
  | .medium => 1
  | .high => 2

/-! ### Constants, mirrored 1:1 from `classify.py` -/

abbrev SCALE : Nat := 100
abbrev LOW_TIER_MAX_FRACTION_NUM : Nat := 35
abbrev MEDIUM_TIER_MAX_FRACTION_NUM : Nat := 50
abbrev RESERVED_OS_UNITS : Nat := 800

/-
The mirror constants are opaque symbols to `omega`: unless they are unfolded,
`SCALE * w` and `LOW_TIER_MAX_FRACTION_NUM * r` are unrelated atoms and the
arithmetic never closes.  Registering them as simp lemmas unfolds them everywhere.
-/
@[simp] theorem SCALE_eq : SCALE = 100 := rfl
@[simp] theorem LOW_TIER_MAX_FRACTION_NUM_eq : LOW_TIER_MAX_FRACTION_NUM = 35 := rfl
@[simp] theorem MEDIUM_TIER_MAX_FRACTION_NUM_eq : MEDIUM_TIER_MAX_FRACTION_NUM = 50 := rfl
@[simp] theorem RESERVED_OS_UNITS_eq : RESERVED_OS_UNITS = 800 := rfl

/-! ### The mathematical predicates the thresholds denote -/

/-- `weight/ram <= 0.35` -/
abbrev LowP (w r : Nat) : Prop := SCALE * w ≤ LOW_TIER_MAX_FRACTION_NUM * r
/-- `weight/ram <= 0.50` -/
abbrev MedP (w r : Nat) : Prop := SCALE * w ≤ MEDIUM_TIER_MAX_FRACTION_NUM * r
/-- `weight/ram > 0.50` -/
abbrev HighP (w r : Nat) : Prop := MEDIUM_TIER_MAX_FRACTION_NUM * r < SCALE * w

/-! ### `resource_tier` -/

/-- Mirror of `resource_tier`.  `r = 0` is the source's `ram_gb <= 0` branch. -/
def resourceTier (w r : Nat) : Tier :=
  if r = 0 then .high else if LowP w r then .low else if MedP w r then .medium else .high

/-- `Tier.rank ∘ resourceTier`, written as a plain `Nat`-valued decision tree. -/
def tierRank (w r : Nat) : Nat :=
  if r = 0 then 2 else if LowP w r then 0 else if MedP w r then 1 else 2

/-! ### Predicate algebra used by the monotonicity proofs -/

theorem lowP_implies_medP {w r : Nat} (h : LowP w r) : MedP w r := by
  have h' : 100 * w ≤ 35 * r := h
  have h2 : 35 * r ≤ MEDIUM_TIER_MAX_FRACTION_NUM * r := by
    simp only [MEDIUM_TIER_MAX_FRACTION_NUM_eq]; omega
  exact Nat.le_trans h' h2

theorem not_medP_iff_highP (w r : Nat) : ¬ MedP w r ↔ HighP w r := by
  constructor
  · intro h
    unfold HighP; unfold MedP at h; omega
  · intro h
    unfold MedP; unfold HighP at h; omega

theorem lowP_of_weight_le {a b r : Nat} (h : a ≤ b) (hb : LowP b r) : LowP a r := by
  have hb' : 100 * b ≤ 35 * r := hb
  have : 100 * a ≤ 100 * b := by omega
  exact Nat.le_trans this hb'

theorem medP_of_weight_le {a b r : Nat} (h : a ≤ b) (hb : MedP b r) : MedP a r := by
  have hb' : 100 * b ≤ 50 * r := hb
  have : 100 * a ≤ 100 * b := by omega
  exact Nat.le_trans this hb'

theorem lowP_of_ram_le {w r₁ r₂ : Nat} (h : r₁ ≤ r₂) (h1 : LowP w r₁) : LowP w r₂ := by
  have h1' : 100 * w ≤ 35 * r₁ := h1
  have h2 : 35 * r₁ ≤ LOW_TIER_MAX_FRACTION_NUM * r₂ := by
    simp only [LOW_TIER_MAX_FRACTION_NUM_eq]; omega
  exact Nat.le_trans h1' h2

theorem medP_of_ram_le {w r₁ r₂ : Nat} (h : r₁ ≤ r₂) (h1 : MedP w r₁) : MedP w r₂ := by
  have h1' : 100 * w ≤ 50 * r₁ := h1
  have h2 : 50 * r₁ ≤ MEDIUM_TIER_MAX_FRACTION_NUM * r₂ := by
    simp only [MEDIUM_TIER_MAX_FRACTION_NUM_eq]; omega
  exact Nat.le_trans h1' h2

/-! ### The returned tier is exactly the predicate it denotes -/

theorem tier_eq_low {w r : Nat} (hr : r ≠ 0) : resourceTier w r = .low ↔ LowP w r := by
  unfold resourceTier
  rw [if_neg hr]
  by_cases h : LowP w r <;> by_cases h2 : MedP w r <;> simp [h, h2]

theorem tier_eq_medium {w r : Nat} (hr : r ≠ 0) :
    resourceTier w r = .medium ↔ ¬ LowP w r ∧ MedP w r := by
  unfold resourceTier
  rw [if_neg hr]
  by_cases h : LowP w r <;> by_cases h2 : MedP w r <;> simp [h, h2]

theorem tier_eq_high {w r : Nat} (hr : r ≠ 0) : resourceTier w r = .high ↔ HighP w r := by
  have hne : resourceTier w r = .high ↔ ¬ MedP w r := by
    unfold resourceTier
    rw [if_neg hr]
    by_cases h1 : LowP w r
    · rw [if_pos h1]
      constructor
      · intro h; exact absurd h (by decide)
      · intro h2; exact absurd (lowP_implies_medP h1) h2
    · rw [if_neg h1]
      by_cases h2 : MedP w r
      · rw [if_pos h2]
        constructor
        · intro h; exact absurd h (by decide)
        · intro h2'; exact absurd h2 h2'
      · rw [if_neg h2]
        exact ⟨fun _ => h2, fun _ => rfl⟩
  exact hne.trans (not_medP_iff_highP w r)

theorem tier_zero_ram (w : Nat) : resourceTier w 0 = .high := by
  unfold resourceTier; simp

/-- The two ways of ranking a tier agree everywhere. -/
theorem tierRank_eq (w r : Nat) : tierRank w r = (resourceTier w r).rank := by
  by_cases hr : r = 0
  · subst hr; simp [tierRank, resourceTier, Tier.rank]
  · rw [tierRank, if_neg hr]
    by_cases h1 : LowP w r
    · rw [if_pos h1, (tier_eq_low hr).mpr h1]; rfl
    · rw [if_neg h1]
      by_cases h2 : MedP w r
      · rw [if_pos h2, (tier_eq_medium hr).mpr ⟨h1, h2⟩]; rfl
      · rw [if_neg h2, (tier_eq_high hr).mpr ((not_medP_iff_highP w r).mp h2)]; rfl

/-- The three tiers are mutually exclusive and exhaustive: every `(w, r)` with
`r ≠ 0` lands in exactly one of them.  This is the "partition" claim, stated
against the *predicates* rather than the return values, so it is contentful. -/
theorem tier_trichotomy (w r : Nat) (hr : r ≠ 0) :
    (resourceTier w r = .low ∧ LowP w r)
      ∨ (resourceTier w r = .medium ∧ ¬ LowP w r ∧ MedP w r)
      ∨ (resourceTier w r = .high ∧ HighP w r) := by
  by_cases h1 : LowP w r
  · exact Or.inl ⟨(tier_eq_low hr).mpr h1, h1⟩
  · by_cases h2 : MedP w r
    · exact Or.inr (Or.inl ⟨(tier_eq_medium hr).mpr ⟨h1, h2⟩, h1, h2⟩)
    · have h3 := (not_medP_iff_highP w r).mp h2
      exact Or.inr (Or.inr ⟨(tier_eq_high hr).mpr h3, h3⟩)

/-! ### Monotonicity: the tier ordering is `.low ≤ .medium ≤ .high` -/

/-- More weight never lowers the tier. -/
theorem tierRank_mono_weight {w₁ w₂ r : Nat} (h : w₁ ≤ w₂) : tierRank w₁ r ≤ tierRank w₂ r := by
  rw [tierRank_eq w₁ r, tierRank_eq w₂ r]
  by_cases hr : r = 0
  · subst hr; rw [tier_zero_ram w₁, tier_zero_ram w₂]; decide
  · rcases tier_trichotomy w₁ r hr with ⟨e1, p1⟩ | ⟨e1, n1, p1⟩ | ⟨e1, p1⟩ <;>
      rcases tier_trichotomy w₂ r hr with ⟨e2, p2⟩ | ⟨e2, n2, p2⟩ | ⟨e2, p2⟩
    · rw [e1, e2]; decide
    · rw [e1, e2]; decide
    · rw [e1, e2]; decide
    · exact absurd (lowP_of_weight_le h p2) n1
    · rw [e1, e2]; decide
    · rw [e1, e2]; decide
    · exact absurd (lowP_implies_medP (lowP_of_weight_le h p2)) ((not_medP_iff_highP w₁ r).mpr p1)
    · exact absurd (medP_of_weight_le h p2) ((not_medP_iff_highP w₁ r).mpr p1)
    · rw [e1, e2]; decide

/-- More RAM never raises the tier. -/
theorem tierRank_anti_ram {w r₁ r₂ : Nat} (h : r₁ ≤ r₂) : tierRank w r₂ ≤ tierRank w r₁ := by
  rw [tierRank_eq w r₂, tierRank_eq w r₁]
  by_cases hr : r₁ = 0
  · subst hr
    rw [tier_zero_ram w]
    cases resourceTier w r₂ <;> decide
  · by_cases hr2 : r₂ = 0
    · subst hr2
      rw [tier_zero_ram w]
      exact absurd (by omega : r₁ = 0) hr
    · rcases tier_trichotomy w r₂ hr2 with ⟨e2, p2⟩ | ⟨e2, n2, p2⟩ | ⟨e2, p2⟩ <;>
        rcases tier_trichotomy w r₁ hr with ⟨e1, p1⟩ | ⟨e1, n1, p1⟩ | ⟨e1, p1⟩
      · rw [e1, e2]; decide
      · rw [e1, e2]; decide
      · rw [e1, e2]; decide
      · exact absurd (lowP_of_ram_le h p1) n2
      · rw [e1, e2]; decide
      · rw [e1, e2]; decide
      · exact absurd (lowP_implies_medP (lowP_of_ram_le h p1)) ((not_medP_iff_highP w r₂).mpr p2)
      · exact absurd (medP_of_ram_le h p1) ((not_medP_iff_highP w r₂).mpr p2)
      · rw [e1, e2]; decide

theorem tier_mono_weight {w₁ w₂ r : Nat} (h : w₁ ≤ w₂) :
    (resourceTier w₁ r).rank ≤ (resourceTier w₂ r).rank := by
  rw [← tierRank_eq w₁ r, ← tierRank_eq w₂ r]
  exact tierRank_mono_weight h

theorem tier_anti_ram {w r₁ r₂ : Nat} (h : r₁ ≤ r₂) :
    (resourceTier w r₂).rank ≤ (resourceTier w r₁).rank := by
  rw [← tierRank_eq w r₂, ← tierRank_eq w r₁]
  exact tierRank_anti_ram h

/-! ### Boundary values -- executed, not asserted -/

/-- 0.35 of 1.00 is `low`: the low/medium boundary is inclusive. -/
theorem boundary_low_inclusive : resourceTier 35 100 = .low := by decide
/-- One unit above it is `medium`. -/
theorem boundary_just_above_low : resourceTier 36 100 = .medium := by decide
/-- 0.50 of 1.00 is `medium`, not `high`: the medium/high boundary is inclusive. -/
theorem boundary_medium_inclusive : resourceTier 50 100 = .medium := by decide
/-- One unit above it is `high`. -/
theorem boundary_just_above_medium : resourceTier 51 100 = .high := by decide

/-! ### `fits_at_all`, and its relationship to the tier test -/

/-- Mirror of `fits_at_all`: `weight_gb + RESERVED_OS_GB < ram_gb`. -/
abbrev fitsAtAll (w r : Nat) : Prop := w + RESERVED_OS_UNITS < r

theorem fits_implies_ram_exceeds_weight {w r : Nat} (h : fitsAtAll w r) : w < r := by
  unfold fitsAtAll RESERVED_OS_UNITS at h; omega

theorem fits_mono_ram {w r₁ r₂ : Nat} (h : fitsAtAll w r₁) (hr : r₁ ≤ r₂) : fitsAtAll w r₂ := by
  unfold fitsAtAll RESERVED_OS_UNITS at *; omega

theorem fits_anti_weight {w₁ w₂ r : Nat} (h : fitsAtAll w₂ r) (hw : w₁ ≤ w₂) : fitsAtAll w₁ r := by
  unfold fitsAtAll RESERVED_OS_UNITS at *; omega

/-- **Redundancy theorem.** On any machine with more than 16 GiB of RAM every
`low`/`medium` model already satisfies `fits_at_all`: since `w <= r/2` and
`r > 1600` units, `w + 800 < r`.  So the `fits_at_all` conjunct in the discovery
filter can only ever change an outcome on machines at or below 16 GiB. -/
theorem fits_redundant_above_16gib {w r : Nat} (h : MedP w r) (hr : 1600 < r) : fitsAtAll w r := by
  unfold MedP at h; unfold fitsAtAll RESERVED_OS_UNITS MEDIUM_TIER_MAX_FRACTION_NUM SCALE at *
  omega

/-- **Disagreement witness.** The two safety notions are genuinely different:
`fits_at_all` means "the weights fit alongside the reserved OS slice", while the
tier test means "the weights are a safe *fraction* of RAM".  A 50 GiB model on
60 GiB of RAM satisfies `fits_at_all` yet is `high`/RISKY. -/
theorem fits_does_not_imply_safe : ∃ w r, fitsAtAll w r ∧ resourceTier w r = .high :=
  ⟨5000, 6000, by decide⟩

end Yojit
