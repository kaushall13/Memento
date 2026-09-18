"""Evaluation suite for the memory system.

Pure, offline metrics (``evals.metrics``) plus the harness that drives them
over a dataset. Metrics are grouped by the question they answer rather than by
where the code lives, so a regression can be attributed to a layer:

- ``retrieval``     — did we find the right memories, ranked well?
- ``answer``        — did the final text use them correctly, and abstain when
                      it had nothing?
- ``temporal``      — is the temporal gate doing real work?
- ``usefulness``    — does memory beat no-memory, and at what token cost?
- ``consolidation`` — is the write path keeping up as the conversation grows?

Every metric is a pure function over primitives (ids, strings, numbers) so it
can be unit-tested without a store, an LLM, or a network.
"""

__all__ = ["metrics"]
