"""Tutorial 36: ViT-base patched batch inference.

Calibration target bucket (batch_inference, small msc, small-medium dps,
small cps, one epoch, tiny samples). One-shot forward pass over a batch of
128 synthetic 224×224 images — covers the vision-classifier path.

Expected runtime: ~10-30 s on any modern GPU.
"""

import asyncio

from krauncher import KrauncherClient

client = KrauncherClient()

HF_MODEL = "hf://models/google/vit-base-patch16-224"


@client.task(
    vram_gb=6,
    timeout=300,
    data_urls=[HF_MODEL],
    dataset_size=1,
    disk_gb=10,
    stream_stderr=True,
)
def vit_batch_inference(batch_size: int = 128, image_size: int = 224):
    """Single-batch ViT forward — synthetic random images, no dataset needed."""
    print("Task started. Importing torch / transformers (~10s)...", flush=True)
    import time

    _t_imp = time.monotonic()
    import torch
    from transformers import ViTForImageClassification, ViTImageProcessor
    print(f"Imports done in {time.monotonic() - _t_imp:.1f}s.", flush=True)

    t0 = time.monotonic()
    model_path = "/data/google__vit-base-patch16-224"

    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(
        model_path, dtype=torch.float16,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded in {time.monotonic() - t0:.1f}s.", flush=True)

    # Synthetic batch: random pixel values in [0, 255], same shape as a real
    # processed batch. Use the processor's normalization for realism.
    import numpy as np
    rng = np.random.default_rng(42)
    raw = rng.integers(0, 256, size=(batch_size, image_size, image_size, 3),
                       dtype=np.uint8)
    images = [raw[i] for i in range(batch_size)]
    inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)

    print(f"Running inference: batch={batch_size}, image={image_size}x{image_size}",
          flush=True)
    t1 = time.monotonic()
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
    infer_sec = time.monotonic() - t1
    print(f"Inference completed in {infer_sec:.3f}s "
          f"({batch_size / infer_sec:.1f} images/s)", flush=True)

    return {
        "batch_size": batch_size,
        "image_size": image_size,
        "infer_sec": round(infer_sec, 4),
        "unique_classes": int(preds.unique().numel()),
    }


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    print("Submitting ViT-base batch inference...")
    handle = await vit_batch_inference()
    print(f"Task submitted: {handle.task_id}")

    result = await handle.wait(timeout=600)
    out = result.output
    print(f"\nBatch: {out['batch_size']}, infer: {out['infer_sec']}s")
    exec_sec = result.execution_time_sec - result.download_sec - result.pip_install_sec
    print(f"Total exec: {exec_sec:.1f}s, Actual CU: {result.actual_cu:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
