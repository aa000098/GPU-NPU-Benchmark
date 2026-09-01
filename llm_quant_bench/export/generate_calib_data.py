"""
Calibration Data Generator
wikitext-2 데이터셋에서 캘리브레이션 샘플을 추출하여
RKNN-LLM (data_quant.json) 및 MNN (calib_data.txt) 포맷으로 저장.

Usage:
    python generate_calib_data.py --output_dir ./calib_data --num_samples 100
"""
import os
import json
import argparse


def load_wikitext2():
    """Load wikitext-2-raw-v1 dataset."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in dataset["text"] if len(t.strip()) > 50]
        return texts
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
        print("Trying local fallback...")
        # Fallback: generate simple calibration data
        return [
            "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings.",
            "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
            "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
            "Natural language processing is a subfield of linguistics and artificial intelligence concerned with interactions.",
        ] * 20


def generate_rkllm_format(texts, num_samples, output_path):
    """Generate RKNN-LLM calibration data (JSON format).
    Format: [{"input": "prompt", "target": "completion"}, ...]
    """
    samples = []
    for i, text in enumerate(texts[:num_samples]):
        text = text.strip()
        if len(text) < 20:
            continue
        # Split text into input (first half) and target (second half)
        mid = len(text) // 2
        input_text = text[:mid]
        target_text = text[mid:]
        samples.append({
            "input": input_text,
            "target": target_text,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[RKLLM] Saved {len(samples)} samples to {output_path}")
    return output_path


def generate_mnn_format(texts, num_samples, output_path):
    """Generate MNN calibration data (plain text, one sample per line)."""
    samples = []
    for text in texts[:num_samples]:
        text = text.strip()
        if len(text) < 20:
            continue
        samples.append(text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s + "\n")

    print(f"[MNN] Saved {len(samples)} samples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate calibration data for quantization")
    parser.add_argument("--output_dir", type=str, default="./calib_data")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    print("Loading wikitext-2 dataset...")
    texts = load_wikitext2()
    print(f"Loaded {len(texts)} text samples")

    # RKNN-LLM format
    rkllm_path = os.path.join(args.output_dir, "data_quant.json")
    generate_rkllm_format(texts, args.num_samples, rkllm_path)

    # MNN format
    mnn_path = os.path.join(args.output_dir, "calib_data.txt")
    generate_mnn_format(texts, args.num_samples, mnn_path)


if __name__ == "__main__":
    main()
