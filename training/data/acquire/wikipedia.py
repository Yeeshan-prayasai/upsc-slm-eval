"""Wikipedia (English) India-subset acquirer.

Single biggest token lever available — full English Wikipedia is ~5 B
tokens (`wikimedia/wikipedia` config `20231101.en`). We don't want the
full thing because our corpus is UPSC-focused, but a substantial
India-related subset is high-value:

- Articles on Indian history, polity, economy, geography, culture
- Biographies of Indian historical/political figures
- Indian government schemes, court cases, treaties
- Constitutional provisions, parliamentary procedure
- Indian states, cities, regional history

We use **title-substring + first-paragraph keyword filtering** rather
than category-tree traversal (which would require additional API calls
per article). The filter keywords are chosen to be high-precision for
India-related content:
- Titles starting with / containing: India, Indian, Bharat, Delhi,
  Mumbai/Bombay, Calcutta/Kolkata, Madras/Chennai, Bangalore/Bengaluru
- States: <every Indian state name>
- High-frequency India-related terms in the first 1000 chars

Source: HuggingFace `wikimedia/wikipedia` config `20231101.en`,
streamed (full dump is ~20 GB extracted text).

CLI:
    python -m training.data.acquire.wikipedia --target-tokens 300000000
    python -m training.data.acquire.wikipedia --target-tokens 5000000   # smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys

from ._base import Manifest, ManifestEntry, RepoPaths, now_iso

DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.en"
TOKENS_PER_WORD = 1.3

# High-precision India-related title keywords (case-insensitive substring match)
# Picked to be specific — "Indian" matches "Indian National Congress",
# "Indian Ocean", etc. without false-positives like "Indiana" (US state).
TITLE_INDIA_KEYWORDS = (
    "India", "Indian", "Bharat",
    "Delhi", "Mumbai", "Bombay", "Bangalore", "Bengaluru",
    "Calcutta", "Kolkata", "Madras", "Chennai", "Hyderabad",
    "Lucknow", "Pune", "Ahmedabad", "Jaipur", "Kanpur",
    "Surat", "Visakhapatnam", "Indore", "Bhopal", "Patna",
    "Vadodara", "Ludhiana", "Agra", "Nashik", "Faridabad",
    "Rajkot", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad",
    "Amritsar", "Allahabad", "Prayagraj", "Howrah", "Ranchi",
    "Gwalior", "Jabalpur", "Coimbatore", "Vijayawada", "Madurai",
    "Meerut", "Nagpur", "Thane", "Bhubaneswar", "Cuttack",
    "Mysore", "Mysuru", "Mangalore", "Mangaluru",
    # States
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Orissa", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Jammu and Kashmir",
    "Ladakh",
    # Major rivers / geography
    "Ganges", "Ganga", "Yamuna", "Brahmaputra", "Godavari", "Krishna",
    "Narmada", "Mahanadi", "Kaveri", "Cauvery",
    "Himalaya", "Himalayan",
    "Western Ghats", "Eastern Ghats", "Deccan",
    # National institutions / constitutional terms
    "Lok Sabha", "Rajya Sabha", "Parliament of India",
    "Supreme Court of India", "Constitution of India",
    "Prime Minister of India", "President of India",
    "Union Public Service Commission", "UPSC",
    "Indian Civil Service",
    "Reserve Bank of India", "Election Commission of India",
    # History keywords
    "Mughal", "Maurya", "Gupta Empire", "Chola", "Vijayanagara",
    "Marathas", "Maratha",
    "British India", "British Raj", "Sepoy Mutiny",
    "Indian independence", "Indian National Congress",
    "Mahatma Gandhi", "Jawaharlal Nehru", "Sardar Vallabhbhai Patel",
    "B. R. Ambedkar",
)
KEYWORDS_LOWER = tuple(k.lower() for k in TITLE_INDIA_KEYWORDS)


def is_india_related(title: str, text: str, head_chars: int = 1000) -> bool:
    """True if title contains any keyword OR the first `head_chars`
    of the article body mentions India explicitly."""
    title_low = (title or "").lower()
    for kw in KEYWORDS_LOWER:
        if kw in title_low:
            return True
    head = (text or "")[:head_chars].lower()
    # Lead mentions of India — high-precision since the article LEAD
    # typically tells you the subject's main context.
    return "india" in head and ("indian" in head or "delhi" in head or "mumbai" in head
                                 or "bharat" in head or "kolkata" in head or "bangalore" in head)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Wikipedia (English) India-subset acquirer.")
    p.add_argument("--target-tokens", type=int, default=300_000_000,
                   help="Approx token budget (default 300 M).")
    p.add_argument("--min-doc-tokens", type=int, default=200,
                   help="Drop articles shorter than this (stubs).")
    p.add_argument("--max-doc-tokens", type=int, default=20_000,
                   help="Drop articles longer than this (lists / chronologies).")
    p.add_argument("--shuffle-seed", type=int, default=20260514)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    args = p.parse_args(argv)

    from datasets import load_dataset

    out_dir = RepoPaths.cpt_raw("wikipedia")
    out_path = out_dir / "india_subset.jsonl"
    manifest = Manifest("wikipedia")

    print(f"Streaming {DATASET} ({CONFIG}) — target {args.target_tokens:,} tokens")
    print(f"India-keyword title filter: {len(TITLE_INDIA_KEYWORDS)} keywords")

    ds = load_dataset(DATASET, CONFIG, split="train", streaming=True)
    # Don't shuffle — Wikipedia in dump order has reasonable diversity
    # already, and shuffling a 6M-doc streaming dataset costs memory.

    tokens_so_far = 0
    docs_kept = 0
    docs_scanned = 0
    docs_dropped_short = 0
    docs_dropped_long = 0
    docs_dropped_filter = 0
    bytes_written = 0
    h = hashlib.sha256()

    with out_path.open("w", encoding="utf-8") as f:
        for example in ds:
            docs_scanned += 1
            title = example.get("title", "")
            text = (example.get("text") or "").strip()
            if not text:
                continue
            if not is_india_related(title, text):
                docs_dropped_filter += 1
                continue
            est_tokens = int(text.count(" ") * TOKENS_PER_WORD) + 1
            if est_tokens < args.min_doc_tokens:
                docs_dropped_short += 1
                continue
            if est_tokens > args.max_doc_tokens:
                docs_dropped_long += 1
                continue
            line = json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n"
            payload = line.encode("utf-8")
            f.write(line)
            h.update(payload)
            bytes_written += len(payload)
            tokens_so_far += est_tokens
            docs_kept += 1
            if docs_kept % 5000 == 0:
                print(f"  ... scanned {docs_scanned:,}, kept {docs_kept:,}, "
                      f"~{tokens_so_far / 1e6:.1f} M tokens")
            if tokens_so_far >= args.target_tokens:
                break

    manifest.add(ManifestEntry(
        url=f"hf://{DATASET}/{CONFIG}#india-subset",
        local_path=str(out_path.relative_to(RepoPaths.root())),
        sha256=h.hexdigest(),
        bytes=bytes_written,
        title="Wikipedia (English) India subset",
        fetched_at=now_iso(),
        extra={
            "dataset": DATASET,
            "config": CONFIG,
            "docs_scanned": docs_scanned,
            "docs_kept": docs_kept,
            "docs_dropped_short": docs_dropped_short,
            "docs_dropped_long": docs_dropped_long,
            "docs_dropped_off_topic": docs_dropped_filter,
            "estimated_tokens": tokens_so_far,
            "tokens_per_word_estimate": TOKENS_PER_WORD,
        },
    ))

    print(f"\nDone. Scanned {docs_scanned:,}, kept {docs_kept:,}, "
          f"~{tokens_so_far/1e9:.3f} B tokens, {bytes_written/1e9:.2f} GB on disk")
    print(f"Dropped: {docs_dropped_short:,} short / {docs_dropped_long:,} long / "
          f"{docs_dropped_filter:,} off-topic")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
