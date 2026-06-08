# Eval Report: ghostbot-eval-core

Run ID: `20260526T143550Z`

- Total: 4
- Passed: 0
- Failed: 4
- Errors: 0
- Pass rate: 0%
- Median latency: 55443 ms
- P90 latency: 67536 ms
- Median tokens: 72150
- P90 tokens: 113572
- Median tool calls: 9
- Median redundant tool rate: 5%
- Requirement hit rate: 25%
- Self-correction rate: 0%
- Recovery cost p50: 0
- Check pass rate: 0%
- Judged scenarios: 3
- Judge pass rate: 0%

| Scenario | Status | Judge | Score | Stop reason | Tool calls | Redundant | Requirement hit | Self-correction | Checks passed | Changed files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plan.shop_improvement.001 | fail | Fail: response was relevant and useful, but it did not include the exact required phrase '优先级', triggering a deterministic failure. | 0.32 | completed | 8 | 0 | 67% | 0% | 0/0 | 0 |
| bugfix.cart_negative_quantity.001 | fail | fail | 0.11 | completed | 10 | 1 | 25% | 0% | 0/1 | 1 |
| tests.cart_regression.001 | fail | fail | 0.00 | completed | 11 | 1 | 25% | 0% | 0/1 | 1 |
| scaffold.empty_checkout.001 | fail | - | - | completed | 4 | 0 | 0% | 0% | 0/0 | 0 |
