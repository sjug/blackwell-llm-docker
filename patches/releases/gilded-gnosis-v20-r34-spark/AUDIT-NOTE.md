# Frozen-composition audit exception (recorded 2026-08-15)

This release directory pins the upstream **gilded-gnosis-v20 r34**
composition exactly as qualified upstream (compose `7302862b`, qualify
`98224d1`), plus the SM121 spark overlay recorded in
`vllm/integration.lock.json` (`spark_overlay` block).

The live repo-wide release-composition audit is RED against current
upstream state and that is EXPECTED here:

- Upstream PRs **#307, #299, #297, #277, #270** postdate the r34 freeze
  and are deliberately unclassified for this release. They belong to the
  audit ledger of whatever release next advances the GG line, not to a
  frozen composition.
- Several excluded PR heads have moved upstream since the freeze. Head
  movement cannot affect this release: both the canonical r34 trees
  (vLLM `4d006a43`, b12x `cd3ce190`, LMCache `9a05c881`) and the spark
  vLLM tree (`c3ffb74f`) are byte-verified against the locks at compose
  time, and the build refuses any tree mismatch.

Scope of validity: this exception covers exactly the
`gilded-gnosis-v20-r34-spark` composition. Re-run and re-classify the
audit before composing any release beyond r34.
