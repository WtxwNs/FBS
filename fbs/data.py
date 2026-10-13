"""Offset-aligned BIOS supervision and local, auditable JSONL preparation.

No corpus, idiom lexicon or task split is silently substituted for paper data.
"""
import argparse
import json
import re
from pathlib import Path


def normalize_prompt(text):
    # Used for split overlap checks; store strings, not opaque hashes.
    return re.sub(r"[\W_]+", "", text.casefold(), flags=re.UNICODE)


def assert_disjoint(train, evaluation):
    held_out = {normalize_prompt(x) for x in evaluation}
    overlap = {normalize_prompt(x) for x in train} & held_out
    if overlap:
        raise ValueError(f"Calibration/evaluation overlap: {len(overlap)} normalized prompts")


def resolve_spans(segments, idioms):
    selected = []
    def overlaps(span):
        return any(span[0] < e and s < span[1] for s, e in selected)
    for span in sorted(set(idioms), key=lambda x: (-(x[1] - x[0]), x[0])):
        if not overlaps(span):
            selected.append(span)
    for span in sorted(set(segments)):
        if not overlaps(span):
            selected.append(span)
    return sorted(selected)


def bios_from_offsets(offsets, spans):
    labels = [2 if end > start else -100 for start, end in offsets]
    for start, end in spans:
        tokens = [i for i, (a, b) in enumerate(offsets) if b > a and start <= a and b <= end]
        if len(tokens) == 1:
            labels[tokens[0]] = 3
        elif tokens:
            labels[tokens[0]] = 0
            for i in tokens[1:]:
                labels[i] = 1
    return labels


def chinese_spans(text, segmenter, lexicon):
    segments, idioms = [], []
    for run in re.finditer(r"[\u4e00-\u9fff]+", text):
        cursor = run.start()
        for word in segmenter.cut(run.group()):
            position = text.find(word, cursor, run.end())
            if position < 0:
                raise ValueError("Segmenter offsets cannot be aligned")
            if 2 <= len(word) <= 6:
                segments.append((position, position + len(word)))
            cursor = position + len(word)
    for idiom in lexicon:
        for match in re.finditer(re.escape(idiom), text):
            idioms.append(match.span())
    return resolve_spans(segments, idioms)


def prepare(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision, use_fast=True, trust_remote_code=False)
    segmenter = None
    lexicon = []
    if args.chinese_bios:
        import pkuseg
        segmenter = pkuseg.pkuseg()
        if not args.idiom_lexicon:
            raise ValueError("Provide the idiom lexicon explicitly")
        lexicon = [s.strip() for s in Path(args.idiom_lexicon).read_text().splitlines() if s.strip()]
    seen = set()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("x") as output, Path(args.input).open() as source:
        for line in source:
            row = json.loads(line)
            text = row["text"]
            normalized = normalize_prompt(text)
            if normalized in seen:
                continue
            seen.add(normalized)
            encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            if len(encoded["input_ids"]) < args.min_tokens:
                continue
            labels = bios_from_offsets(encoded["offset_mapping"], chinese_spans(text, segmenter, lexicon)) if segmenter else [-100] * len(encoded["input_ids"])
            if segmenter:
                # Paper: no externally supplied chunk labels for English.
                labels = [label if re.search(r"[\u4e00-\u9fff]", text[a:b]) else -100
                          for label, (a, b) in zip(labels, encoded["offset_mapping"])]
            record = {"input_ids": encoded["input_ids"], "bios_labels": labels,
                      "source_id": row.get("source_id"), "tokenizer": args.tokenizer, "revision": args.revision}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Prepared {count} documents. Deduplication is exact normalized text; paper similarity>0.7 filtering requires its original specification.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL records with text and source_id")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--min-tokens", type=int, default=512)
    parser.add_argument("--chinese-bios", action="store_true")
    parser.add_argument("--idiom-lexicon")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
