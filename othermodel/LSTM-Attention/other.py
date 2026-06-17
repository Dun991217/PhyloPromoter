import os
from pathlib import Path


def read_fasta(path):
    records = []
    header = None
    seq_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            records.append((header, "".join(seq_lines)))

    return records


def write_fasta(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for header, seq in records:
            f.write(f"{header}\n")
            f.write(f"{seq}\n")


def center_crop_sequence(seq, target_len=100):
    seq = seq.strip().upper()
    n = len(seq)
    if n < target_len:
        raise ValueError(f"Sequence length {n} is shorter than target length {target_len}")

    start = (n - target_len) // 2
    end = start + target_len
    return seq[start:end]


def crop_fasta_center(input_fasta, output_fasta, target_len=100):
    records = read_fasta(input_fasta)

    cropped_records = []
    lengths = set()

    for header, seq in records:
        cropped_seq = center_crop_sequence(seq, target_len=target_len)
        cropped_records.append((header, cropped_seq))
        lengths.add(len(seq))

    write_fasta(cropped_records, output_fasta)

    print(f"Input : {input_fasta}")
    print(f"Output: {output_fasta}")
    print(f"Total sequences: {len(records)}")
    print(f"Original lengths: {sorted(lengths)}")
    print(f"Cropped length : {target_len}")
    print("-" * 60)


if __name__ == "__main__":
    file_pairs = [
        (
            "other_model/Promoter/zou/project/dataset/human_deepromoter_dataset_300/fa/positive.fa",
            "other_model/Promoter/zou/project/dataset/human_deepromoter_dataset_100/fa/positive.fa",
        ),
        (
            "other_model/Promoter/zou/project/dataset/human_deepromoter_dataset_300/fa/negative.fa",
            "other_model/Promoter/zou/project/dataset/human_deepromoter_dataset_100/fa/negative.fa",
        ),
        (
            "other_model/Promoter/zou/project/dataset/mouse_deepromoter_dataset_300/fa/positive.fa",
            "other_model/Promoter/zou/project/dataset/mouse_deepromoter_dataset_100/fa/positive.fa",
        ),
        (
            "other_model/Promoter/zou/project/dataset/mouse_deepromoter_dataset_300/fa/negative.fa",
            "other_model/Promoter/zou/project/dataset/mouse_deepromoter_dataset_100/fa/negative.fa",
        ),
    ]

    for input_fa, output_fa in file_pairs:
        Path(output_fa).parent.mkdir(parents=True, exist_ok=True)
        crop_fasta_center(input_fa, output_fa, target_len=100)