# Data preparation protocol

The current baseline accepts UTF-8 text files with one complete document per non-empty line. JSONL datasets must first be exported by selecting the text field into this format; the exporter should record its source dataset, revision, configuration, license, selection rule and export code version.

Run `frontier prepare-data --input SOURCE.txt --output data/tinystories --validation-fraction 0.02 --seed 17`. The splitter groups identical documents before assignment so identical rows cannot fall across the train/validation boundary. It writes `train.txt`, `validation.txt` and `data_manifest.json`, which records source and split SHA-256 hashes. Training refuses exact train/validation overlap and checks the split hashes whenever a run loads the corpus.

The experiment configs refer to `data/tinystories` as a location only. No dataset is bundled or assumed licensed by this repository. Confirm the source license and usage terms, and use the same exported files for every control and candidate. Keep the data outside Git; record the manifest with each run.

For early infrastructure checks, `configs/smoke.json` creates a fixed synthetic fixture automatically. Its scores are not downstream language-capability evidence.


## Reproducing the bounded TinyStories subset

The M1 corpus run uses bytes 0–20,971,519 of `TinyStories-train.txt` at the pinned `roneneldan/TinyStories` revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. The [dataset card](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md) declares `cdla-sharing-1.0` and describes the dataset as synthetic short stories. The experiment deterministically holds out 2% of selected documents from this training prefix; it does not use the dataset's separate validation file. This bounded split verifies training and evaluation plumbing, not broad capability.

`scripts/prepare_tinystories_prefix.py` splits on `<|endoftext|>`, discards the last partial story if the byte range cuts through one, and collapses story whitespace to single spaces. It writes one document per line and a provenance sidecar containing the dataset revision, source file, license, byte range, upstream size, raw-prefix SHA-256, complete-story count, discarded-tail size, and text transformation. Pass that sidecar to `frontier prepare-data --source-metadata`; the resulting split manifest is copied into every run. Raw and prepared corpus files remain local under the root `data/` directory, which is git-ignored.