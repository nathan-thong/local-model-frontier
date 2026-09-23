# Tracked experiment artifacts

`m2-gqa-002/` is the sanitized evidence bundle for the corrected three-seed grouped-query attention comparison. The experiment log in [`../EXPERIMENTS.md`](../EXPERIMENTS.md) gives the measured results and their synthetic TinyStories scope.

Verify the bundle from the repository root:

```powershell
frontier verify-evidence --bundle-dir results/m2-gqa-002
```

The bundle contains six sanitized run configurations, a sanitized corpus manifest, the comparison and sweep-plan summaries, and an inventory of the source artifact hashes and availability. It excludes corpus documents, tokenizer files, checkpoints, and model weights. The per-run records identify which excluded artifacts were available and provide their hashes where available.

The manifest checks each included file against its recorded size and SHA-256 and rejects missing or unlisted files. For an external integrity check, compare the reported bundle SHA-256 with the value recorded in [`../EXPERIMENTS.md`](../EXPERIMENTS.md); the bundle does not carry a digital signature. Training reproduction still requires the exact prepared corpus and tokenizer identified by the hashes in the bundle.
