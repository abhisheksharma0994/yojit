import Yojit.Tier
import Yojit.Limits
import Yojit.Kv
import Yojit.Tuning

/-!
  Axiom audit.  Every claim in the write-up must rest on Lean's three standard
  axioms (`propext`, `Classical.choice`, `Quot.sound`) and *nothing else* -- in
  particular no `sorryAx`, which is what a placeholder proof would introduce.
  `lake build` prints these; the expected output is one of:
    'Yojit.<name>' does not depend on any axioms
    'Yojit.<name>' depends on axioms: [propext, Classical.choice, Quot.sound]
-/

#print axioms Yojit.tier_trichotomy
#print axioms Yojit.tier_mono_weight
#print axioms Yojit.tier_anti_ram
#print axioms Yojit.fits_redundant_above_16gib
#print axioms Yojit.fits_does_not_imply_safe
#print axioms Yojit.min_context_is_not_a_floor
#print axioms Yojit.outputOf_le_quarter
#print axioms Yojit.outputOf_le_self
#print axioms Yojit.output_ratio_holds_below_min_context
#print axioms Yojit.legacy_output_exceeds_quarter_of_lt
#print axioms Yojit.legacy_output_exceeds_quarter_at_2048
#print axioms Yojit.context_bounds
#print axioms Yojit.context_dvd
#print axioms Yojit.kvBits_overshoot
#print axioms Yojit.kvStart_lt_requested_context_of_not_fp16_fits
#print axioms Yojit.kv_overshoot_at_project_test_parameters
#print axioms Yojit.kvPlanContext_le_requested
#print axioms Yojit.kvPlanContext_eq_requested_of_fits
#print axioms Yojit.kvPlanContext_fits_headroom
#print axioms Yojit.kvStartClamped_le_context
#print axioms Yojit.kvBytesPerToken_le_self
#print axioms Yojit.kvMaxTokens_ge_div
#print axioms Yojit.plan_keeps_the_estimate
#print axioms Yojit.plan_at_project_test_parameters
#print axioms Yojit.plan_fits_at_project_test_parameters
#print axioms Yojit.legacy_plan_overshoots_at_project_test_parameters
#print axioms Yojit.tierIndex_lt_table_length
#print axioms Yojit.tables_match_bucket_count
#print axioms Yojit.promptCacheUnits_bounds
#print axioms Yojit.threads_le_cores
