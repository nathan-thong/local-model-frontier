# Paired sweep planning

`frontier plan-sweep` writes a hash-protected, immutable plan for a small seed-paired experiment. It resolves the configs and output directories, calculates each arm's planned optimizer steps, tokens and versioned FLOP estimate, records the available host configuration, and rejects a plan that exceeds its declared total compute cap. It does not instantiate a model, start a training run, download data or overwrite an earlier plan.

Use a JSON spec with arm-config and output-root paths relative to that spec file. A `data_dir` inside a run config follows the training CLI convention and resolves relative to the current working directory:

```json
{
  "schema_version": 1,
  "name": "m2-gqa-pilot",
  "baseline": {"id": "mha", "config": "baseline.json"},
  "candidates": [{"id": "gqa", "config": "gqa.json"}],
  "seeds": [17, 42, 123],
  "output_root": "../../runs/m2-gqa-pilot",
  "total_compute_cap_flops": 100000000000000
}
```

Then run:

```powershell
frontier plan-sweep --spec experiments/m2-gqa-pilot.json --output runs/plans/m2-gqa-pilot.json
```

The plan records whether each candidate's configured per-seed FLOP estimate is within 1% of its baseline. This is a warning/diagnostic field, not a claim of matched work. Review the plan before running any job. Every job contains its full resolved config and a hash; the overall digest changes if any job is edited. Plans with unsupported sequence-module compute are rejected until an estimator is registered.

Execute a reviewed plan serially with:

```powershell
frontier execute-sweep --plan runs/plans/m2-gqa-pilot.json --max-runtime-seconds 900
```

The executor writes a sibling `.status.json` atomically and takes a `.lock` file to prevent two processes from launching the same sweep. It skips completed jobs only after verifying their config, token count, compute estimate and planned hardware fields. It resumes a partial job only when the saved summary is `running`, the original config matches, and `checkpoints/last.pt` exists. The optional runtime budget is enforced at optimizer-step boundaries; the current step and checkpoint write can extend slightly past the requested duration. The run is then left resumable. The budget applies to one executor invocation; a later resume is a new invocation with its own explicit limit. It stops at the first failed or incompatible job and records the reason. It does not retry failed jobs automatically. If a process is forcibly terminated, inspect the lock and status before removing a stale lock and resuming.

The compute cap is the sum of planned per-job training estimates; actual completed jobs are checked against those budgets. The optional wall-time cap is per execution invocation, not cumulative across later resumes. These commands only operate on local run directories. Do not treat the plan or its compute estimates as training results; compare completed runs with `frontier compare-seeds`, which still enforces provenance, paired seeds and compute eligibility.
