# Design decisions

1. **Tools over free-form SQL.** Typed tools constrain what the model can do, make results reproducible and testable, and keep prompts small. `run_sql` exists only as a guarded last resort.
2. **Deterministic numbers, LLM narrative.** Statistics are computed in Python/SQL; the model selects tools and explains. The eval harness checks that numbers in answers exist in tool outputs.
3. **Errors are data.** Tool failures (unknown drug, bad args) return `is_error` results so the model can recover (e.g. call `search_drugs`) instead of crashing the loop.
4. **Bounded and observable.** `PV_MAX_STEPS` caps the loop; every call records name, input, output, latency; token usage is aggregated.
5. **Defence in depth for SQL.** Validator, row cap, read-only URI connection, authorizer allow-list, interrupt-based timeout.
6. **Reproducible demo.** Seeded synthetic data with known planted signals lets tests assert that the pipeline rediscovers them.

## Signal method
For each reaction, a 2x2 table (drug vs. all other cases) gives PRR and ROR (+95% CI) and chi-square. A signal is flagged when cases >= 3, PRR >= 2 and chi-square >= 4 (Evans). Results are ranked by flag, then lower bound of the ROR CI.
