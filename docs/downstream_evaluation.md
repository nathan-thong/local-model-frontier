# Downstream evaluation adapter

The initial adapter accepts UTF-8 JSONL with one example per line:

```json
{"prompt":"The passage says that","target":" the sky is blue."}
```

Every example must contain string `prompt` and `target` fields. The evaluator reports greedy exact match and teacher-forced conditional NLL, scores target tokens only, appends EOS to the target, and saves per-example predictions in `evaluation.json`. Generation is capped by the model's configured maximum sequence length; prompts that leave too little room for the requested generation fail with an explicit error.

This is a generic task interface, not a named benchmark. For a real benchmark, add a versioned adapter that exports deterministic prompt/target rows, records source revision and scoring rules, and preserves per-example outputs. Avoid changing prompt templates or examples between control and candidate runs. Do not interpret byte-token exact match as a general capability score without a task-specific justification.
