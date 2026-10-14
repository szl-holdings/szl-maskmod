# Changelog

## 0.2.0 (2026-10-14)

### Changed (semantics)
- Keep-masks are hard constraints. `maskmod_attn` now re-applies the causal mask and the
  boolean `block_mask` after `score_mod` runs, so an excluded position stays at `-inf` and
  gets attention weight exactly 0 whatever the callback returns (finite, clamped or NaN).
  Before 0.2.0 a `score_mod` that returned finite scores at masked positions restored
  attention there. Callbacks that already preserve `-inf` produce identical output.
- A fully excluded query row still yields a zero output row.
- The test oracle in `tests/test_block_mask.py` follows the new order, and adversarial
  `score_mod` regressions check against an independent row-wise softmax oracle.

## 0.1.0
- Initial release.
