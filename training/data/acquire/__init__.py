"""Per-source corpus acquirers.

Each module exposes a `main(argv)` CLI and writes to
`data/cpt_raw/<source>/` plus a per-source `manifest.json` containing
URL, local path, SHA-256, byte count, fetch timestamp, and title.

Acquirers produce ONLY raw text/PDFs. Cleaning, deduplication, OCR,
and tokenization happen in `training.data.build_cpt_corpus`.
"""
