# Formal verification of `yojit`

Mechanical verification of `src/yojit/classify.py` (Lean 4) and of the
concurrency in `opencode_sync.py` / `server.py` / `locking.py` (TLA+ / TLC).

Base revision: `github.com/abhisheksharma0994/yojit` at `d06831f` (2026-08-29).
Toolchain: Lean 4.34.0 (no Mathlib), and TLC from the `tla2tools.jar` asset on the
TLA+ `v1.8.0` release — whose bytes are pinned by SHA-256 in
`tools/fetch-tla2tools.sh`, because TLC's self-reported version is a date stamp
rather than a release number. Python suite at this revision: 362 passed, 99.5%
line coverage.

This directory started as an audit and is now a **regression guard**. The audit
found three defects, all of which are fixed in the same change that updated the
models; the models are what stop the fixes from silently drifting back, because
they are executed by `pytest` and by CI (see "How the guard is wired in").

## What each tool is for, and why

`classify.py` is pure arithmetic, and the properties worth wanting are
universally quantified ("for every model and every machine"), so Lean is right:
it proves statements about *all* inputs, not samples.

The state writes are effectful and concurrent, so Lean would only produce a
model. The properties worth wanting there are about *interleavings* — two
`yojit` processes doing read-modify-write on one file, or racing for one port —
so TLA+ + TLC is right: it explores the reachable state space and returns a
counterexample trace, or exhausts it.

Note the division of labour is not "custom logic vs. IO". It is "what shape is
the property": pure function + universal quantifier → Lean; interleaving →
TLA+; network/subprocess orchestration → neither, and tests are the honest tool.

## Reproducing

```bash
# Everything, through the test suite (skips cleanly if a toolchain is missing):
cd .. && pytest tests/test_formal_models.py -v

# Or with failures instead of skips, which is what CI does:
YOJIT_REQUIRE_FORMAL=1 pytest tests/test_formal_models.py -v

# Lean alone: 108 theorems, no sorry, no extra axioms
cd formal/lean
lake build                      # prints the axiom audit from Yojit/Checks.lean

# TLA+ alone. Note ServeLifecycleConsistency.cfg checks ServeLifecycle.tla --
# a .cfg names a SPECIFICATION, not necessarily a file of the same name.
cd ../tla
JAR=../tools/tla2tools.jar
java -jar $JAR -deadlock -config OpencodeSync.cfg OpencodeSync.tla
java -jar $JAR -deadlock -config OpencodeSyncLocked.cfg OpencodeSyncLocked.tla
java -jar $JAR -deadlock -config ServeLifecycle.cfg ServeLifecycle.tla
java -jar $JAR -deadlock -config ServeLifecycleConsistency.cfg ServeLifecycle.tla
java -jar $JAR -deadlock -config ServeLifecycleRecord.cfg ServeLifecycleRecord.tla
```

`Yojit/Checks.lean` prints `#print axioms` for 27 headline theorems. Every one
reports either "does not depend on any axioms" or `[propext, Quot.sound]` /
`[propext, Classical.choice, Quot.sound]`. No `sorryAx`, which is what a
placeholder proof would introduce.

## How the guard is wired in

| Mechanism | What it stops |
|---|---|
| `tests/test_formal_models.py` | the models rotting unexecuted: `lake build` must be clean, all five TLC runs must reach their expected verdict |
| `test_lean_proofs_contain_no_sorry` | a placeholder proof turning the theorems below it into unchecked claims |
| `test_lean_axiom_audit_is_complete_and_clean` | a theorem resting on something beyond Lean's three standard axioms |
| `test_lean_constants_match_the_python_source` | the model drifting from the code. It derives each Lean constant *from the Python attribute* and compares, including unit changes (`RESERVED_OS_GB = 8.0` → `RESERVED_OS_UNITS = 800`) and inversions (`SAFETY_FACTOR = 0.25` → `SAFETY_FACTOR_DENOM = 4`) |
| `test_tlc_configs_check_the_invariants_they_claim_to` | a `.cfg` silently dropping its `INVARIANT` line, which would make its run pass while checking nothing |
| `.github/workflows/formal.yml` | all of the above going quiet in CI: it sets `YOJIT_REQUIRE_FORMAL=1`, so a missing toolchain is a failure, not a skip |

Both toolchains stay optional for contributors — without elan or Java the tests
skip with a message. The one test that makes the Lean theorems mean anything
about shipped code is the constant mirror; the rest are checks on the models.

## Proved to hold (Lean, for all inputs)

| Theorem | Statement |
|---|---|
| `tier_trichotomy` | `low`/`medium`/`high` partition the space exactly |
| `tier_mono_weight`, `tier_anti_ram` | more weight never lowers the tier; more RAM never raises it |
| `boundary_low_inclusive`, `boundary_medium_inclusive` | the 0.35 and 0.50 boundaries are inclusive (`<=`, not `<`) |
| `context_bounds` | `min(native, 4096) <= context <= min(native, 65536)` exactly |
| `context_dvd` | `context >= 4096` implies `context` is a multiple of 4096 |
| `context_mono_ram`, `context_anti_weight` | `context` is monotone in RAM, anti-monotone in weight |
| `min_context_is_not_a_floor` | a native context below 4096 comes back unchanged |
| `outputOf_le_quarter` | `output <= context / 4` for every context that holds a token |
| `outputOf_le_self`, `outputOf_pos` | `output` never exceeds the window, and is never zero |
| `output_ratio_holds_below_min_context` | and both hold below `MIN_CONTEXT` too — the fix |
| `legacy_output_exceeds_quarter_of_lt` | the pre-fix floor exceeded a quarter for *every* input below `MIN_CONTEXT`, so the two formulas are proved apart, not merely replaced |
| `kvPlanContext_le_requested` | the KV plan never raises the requested context |
| `kvPlanContext_eq_requested_of_fits` | when the request fits, the plan is the identity |
| `kvPlanContext_fits_headroom` | whenever one token fits, the reported context fits the headroom |
| `kvStartClamped_le_context` | `quantized_kv_start` is a valid index into the window launched with |
| `plan_at_project_test_parameters`, `plan_fits_at_project_test_parameters` | the project's own test parameters now yield a context that fits, with `legacy_plan_overshoots_at_project_test_parameters` proved as the contrast |
| `plan_keeps_the_estimate` | the estimate's own context fits the budget that chose it — no shrink, no rounding — unless memory cannot afford `MIN_CONTEXT` at all |
| `kvMaxTokens_ge_div`, `kvBytesPerToken_le_self` | a quantized cache is never narrower than the unquantized bound it was chosen from |
| `kvBits_mem`, `kvBits_mono_headroom` | KV quantization is one of 16/8/4, monotone in headroom |
| `kvStart_lt_requested_context_of_not_fp16_fits` | when fp16 does not fit, the start index is below the requested context |
| `tierIndex_le_four`, `tierIndex_lt_table_length` | the tuning tables are never indexed out of bounds |
| `tierIndex_mono`, `prefillStepSize_mono` | chunking is monotone in RAM headroom |
| `promptCacheUnits_bounds` | `--prompt-cache-bytes` stays within its 0.5-8 GiB window |
| `threads_ge_one`, `threads_le_cores` | one core is left for the OS, single-core still gets one |
| `fits_redundant_above_16gib` | above 16 GiB, `low`/`medium` already implies `fits_at_all` |
| `fits_does_not_imply_safe` | `fits_at_all` and the tier test are genuinely different notions |

The project already tests several of these at two or four sample points
(`test_resource_tier_boundaries`, `test_compute_launch_tuning_*`). The Lean
theorems hold at every point, including ones no test covers.

## The three defects the audit found, and their fixes

### 1. KV-cache quantization could exceed its own headroom

`default_kv_cache_overrides` selected `16` if it fit, else `8`, else `4` — and
returned `4` whether or not it fit (`kvBits_overshoot`). No branch reported a
shortfall, so the caller could not tell "the most precise width that fits" from
"nothing fits".

At the parameters of the project's *own* test
(`test_default_kv_cache_overrides_falls_back_to_4bit_when_even_8bit_does_not_fit`:
`f = 131072`, `context = 16384`, headroom `268435456` bytes), the 4-bit cache
needs `16384 * 131072 / 4 = 536870912` bytes — **twice** the available headroom —
and the test asserted `"4"` as correct.

**Fix.** `resolve_kv_cache` returns a `KvPlan` carrying `fits`, the effective
context, and the arithmetic it used. When even the lowest width does not fit,
the context comes down to what does (`8192` at those parameters, i.e.
`8192 * 32768 = 268435456`, exactly the headroom), the caller prints what
happened, `output` is recomputed from the *effective* context, and
`kv_fit_limits` applies the same rule at install time so the manifest never
records a window this machine cannot serve. Guarded by
`test_resolve_kv_cache_shrinks_the_context_when_even_4bit_does_not_fit` and the
Lean `kvPlanContext_fits_headroom`.

### 2. `output` was not a quarter of `context` below `MIN_CONTEXT`

`MIN_CONTEXT = 4096` is a floor on the *memory-derived estimate only*; a model
whose native context is lower keeps that native value
(`min_context_is_not_a_floor`). That part is intended and stays.

The consequence was not: `output = max(MIN_OUTPUT, min(context // 4, 4096))`,
and `MIN_OUTPUT` is `MIN_CONTEXT / 4 = 1024`, so for that 2048-token model
`output` was pinned to 1024 — **half** the window. That value is what
`opencode_sync` writes into `limit.output`, so the client was told it could
generate 1024 tokens into a 2048-token window.

**Fix.** `output_for_context(context) = max(1, min(context // 4, 4096))` — a
strict quarter with no floor above 1, used by every caller so the stored value
and the serve-time value cannot disagree. `2048 → 512`. The old formula and the
new one are both theorems, so the difference is checked rather than remembered:
`legacy_output_exceeds_quarter_of_lt` versus `output_ratio_holds_below_min_context`.

### 3. The state writes had no lock (TLA+, three counterexamples)

**Lost update** (`OpencodeSync`, invariant `SyncComplete`, 27 states generated /
21 distinct). `sync()` reads the whole file, edits its copy, and writes the whole
file back, so two concurrent writers lose one another's models:

```
Read(1) Mutate(1) | Read(2) Write(1) Mutate(2) Write(2)   =>  file = {0, 2}   (model 1 lost)
```

**A reported server could already be dead** (`ServeLifecycle`, invariant
`ReportedServerIsAlive`, 56 states / 43 distinct). `_free_port(PORT)` killed
whatever held the port, with no ownership check and no cooperative lock, so
process 1 could print `Server running in the background (PID ...)` and hand off
to opencode bound to its model, and process 2 then killed that server.

**`(running)` could be advertised for a dead server**
(`ServeLifecycleConsistency`, invariant `DefaultMatchesAdvertised`, 96 states /
68 distinct). The manifest default and the `(running)` marker in `opencode.json`
were two separate unlocked read-modify-writes, so they could disagree.

**Fix.** `locking.py` provides one advisory lock (exclusively-created file:
portable, no daemon, and a stale lock is broken when its recorded PID is gone)
plus an atomic write via temp-file-and-rename. Every mutation of `manifest.json`
and `opencode.json` takes it; `set_default` and `sync` share one critical
section; `server.json` records which pid owns the port so `_free_port`,
`yojit stop`, and `yojit status` only touch a server yojit actually started,
raising instead of killing a stranger's process on 8080.

`OpencodeSyncLocked.tla` repeats the lost-update model with a mutex held across
the whole read-modify-write and has **no** violation (61 states generated, 61
distinct, `N = 3`). `ServeLifecycleRecord.tla` models the fixed lifecycle —
ownership record, launch lock, single commit section, `stop`/`status` reading the
record — and all four invariants hold (`ForeignListenerNeverKilled`,
`DefaultMatchesAdvertised`, `StatusNeverReportsForeign`,
`StatusOnlyReportsLiveServers`; 49 states generated, 16 distinct, `N = 4`).
`DefaultMatchesAdvertised` is deliberately the same invariant name that the
pre-fix spec violates.

## Three things the work on the fixes turned up

### The `quantized_kv_start` clamp is *not* provably dead (corrected)

The first pass concluded the `min(start, context)` clamp was dead code, on the
grounds that emitting `quantized_kv_start` requires fp16 not to fit, which
already forces `start < requested context`
(`kvStart_lt_requested_context_of_not_fp16_fits`).

That argument is about the *requested* context, not the one launched with. Now
that the shrink exists, the emitted window is
`round_to_4096(min(context, max_tokens))`, and a round-**down** applied to a
bound derived from different arithmetic is exactly the kind of step where an
index computed elsewhere can land past the end. The clamp is kept, and the
guarantee is stated as a theorem in its own right
(`kvStartClamped_le_context`) rather than inferred from the fp16 argument. The
`min(start, effective_context)` in `classify.py` carries this reasoning in a
comment.

### The install estimate and the KV fit check read different headroom

Found by CI's `e2e` job, and the only defect in this document that reached a
user-visible failure.

`estimate_limits_from_config` sized the context against
`max(ram - weight - RESERVED_OS_GB, MIN_HEADROOM_GB)` — a 1.0 GiB floor — while
`resolve_kv_cache` checked that context against its own `max(..., 0.1)` floor. On
any machine whose RAM does not cover the weights plus the 8 GiB OS reservation
those floors differ tenfold, so the check rejected the estimate *by
construction* and the new shrink fired on every install.

On the CI runner — 7 GB serving a 0.6 GB model, `f = 12288` bytes per token —
the estimate chose 20480 tokens and the fit check cut the launched window to
8192: `max(7 - 0.6 - 8, 0.1) * 2^30 * 0.25` is 26.8 MB, which is 8738 tokens at
4 bits, rounded down to 8192. opencode's own request needs 9620 (7572 of system
prompt plus a 2048 output budget), so every prompt came back `400 Bad Request:
MAX_KV_SIZE is 8192` — a server that starts, reports itself healthy, and cannot
answer a single question.

**Fix.** One budget function, `headroom_bytes()`, read by both sizing decisions,
with the 1.0 GiB floor as the only floor. `Kv.lean` now proves the property the
split could not state: `plan_keeps_the_estimate` says handing the plan the
estimate's own context is the identity, and its single hypothesis is the one case
where a shrink is legitimate — memory that cannot afford `MIN_CONTEXT` at all,
where the floor rather than memory set the context.
`tests/test_classify.py` pins the same property for the real LFM2.5-1.2B config at
7/8/8.6 GB, and the general invariant across a 131072- and a 12288-byte-per-token
model. Both fail against the pre-fix code.

### A stale lock that cannot be deleted used to spin forever

Found by `test_a_stale_lock_that_cannot_be_removed_still_times_out`, which hung
the suite: in `state_lock`'s retry loop the "break the stale lock" path
`continue`d past the deadline check, so a stale lock in a read-only directory
looped without ever timing out. The deadline is now checked on every path. This
is a fix to code written in this same change, and it exists because the test was
written to fail rather than to pass.

## Remaining gaps

**Documentation claims the code does not meet.**

- The README says "every knob beyond context/output is recomputed fresh from real
  RAM headroom and CPU core count on every `serve` call, never a fixed constant".
  `ngl = 999`, `decode_concurrency = 1`, and `prompt_concurrency = 1` are literal
  constants (`ngl_is_not_spec_derived`). The source comments say this is
  deliberate, so the README sentence is the thing that overstates.
- The README's "~50% dividing line" describes the tier test. `fits_at_all` uses a
  different notion (a flat 8 GiB reserve), and the two disagree:
  `fits_does_not_imply_safe` exhibits a 50 GiB model on 60 GiB of RAM that passes
  `fits_at_all` and is still `high`/RISKY.

**A window the fix narrows but cannot close.** `serve()` releases the launch lock
before it publishes the handoff, so a second `yojit serve` can still replace the
server in between. `_publish_handoff` re-validates the record inside the same
critical section that writes the claim, and refuses to launch opencode if the
server is no longer ours
(`test_serve_refuses_to_hand_off_after_another_process_replaced_the_server`).
What remains is the unavoidable gap between that check and `exec`, and the
by-design case of a user deliberately starting a second model — which kills the
first server, because one port means one server.

**A latent maintenance hazard.** `_HEADROOM_TIER_GB` has 4 ceilings and the three
tuning tables have 5 entries (`tables_match_bucket_count`). Add a fifth ceiling
and Python raises `IndexError` during `serve`; here
`tierIndex_lt_table_length` simply stops being provable, so the build breaks
instead of the user's launch.

## What this does **not** establish

- **Floats.** `classify.py` works in IEEE-754; the Lean model works in exact
  integers. The models clear denominators (`100 * w <= 35 * r`) and replace
  `headroom_gb * 2**30 * 0.25` with a single truncating division (`* 2^30 / 400`).
  The bounds, monotonicity, and the KV fit arithmetic are insensitive to that
  difference, but a theorem about float rounding does not exist here.
- **Spec-to-code correspondence.** Lean proves statements about the Lean model of
  `classify.py`; TLA+ checks a hand-written abstraction of the state writes. That
  each faithfully represents the Python is asserted, not proved. The constant
  mirror test makes that assertion mechanical for the numbers, which is the part
  that actually drifts — but it is a test, not a proof.
- **Model checking is not proof.** TLC's results are "no counterexample in the
  explored state space" (state counts above, and the model is small by design —
  the point is the ownership and commit logic, not the load), not "no
  counterexample exists".
- **The effectful surface.** GGUF header parsing, filesystem layout, subprocess
  management, and Hugging Face network calls are untouched by everything here.
  Those want integration tests and fault injection.
- **Liveness.** Nothing here is checked under fairness: "a serve that is retried
  eventually succeeds" is not a property any of these specs state.
- **Client prompt size.** Nothing relates the window yojit picks to the size of
  the request a real client sends. The shrink that broke `e2e` was arithmetically
  correct against the budget it read; what no arithmetic here could know is that
  opencode's request is 9620 tokens. A unit test can pin the numbers, but only
  the `e2e` job puts a real prompt through a real server.

## Layout

```
formal/
  lean/
    lakefile.toml, lean-toolchain
    Yojit.lean            -- imports Checks
    Yojit/Tier.lean       -- resource_tier, fits_at_all
    Yojit/Limits.lean     -- estimate_limits_from_config, output_for_context (+ the legacy formula)
    Yojit/Kv.lean         -- kv selection, then the KvPlan layer that fixes it
    Yojit/Tuning.lean     -- compute_launch_tuning
    Yojit/Checks.lean     -- axiom audit
  tla/
    OpencodeSync.tla/.cfg          -- the lost update
    OpencodeSyncLocked.tla/.cfg    -- the fix, checked
    ServeLifecycle.tla             -- the pre-fix lifecycle, two invariants in two .cfg files
    ServeLifecycleRecord.tla/.cfg  -- the fixed lifecycle, four invariants
  tools/fetch-tla2tools.sh   -- pinned, SHA-256-verified download of the jar
```

`tests/test_formal_models.py` runs all of the above; `.github/workflows/formal.yml`
runs it with skips disabled.
