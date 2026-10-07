"""Tutorial 55: The assay of a task and its compute coefficient on every GPU — before running it.

Nothing runs on a GPU here. The analyzer reads your code and issues an assay:
how much work the payload is, measured on the reference card, in CU. The assay
is a document — keep it. Given an assay, the ladder gives every GPU's
coefficient for the compute phase against the reference card, fastest first.

    Client                                   Analyzer / broker
    ──────                                   ─────────────────
    analyzer.classify(source)           →   analysis (code E2E-encrypted)
    analyzer.assay(job_id)              ←   assay v1: work on the reference card
    client.ladder(assay)                ←   compute coefficient per GPU (broker)

Time on a GPU = compute_ratio x the assay's compute phase + its other phases
(warmup, I/O, setup), which do not depend on the GPU. Multiply by the price
you pay.

The ladder is available to accounts with ladder access enabled. The assay
stays valid until the analyzer is recalibrated; after that client.ladder()
raises AssayOutdated — analyze the code again for a fresh assay.

Run it from a folder holding your .env (CAS_API_KEY):

    python tutorial/55_assay_ladder.py
"""

import asyncio
import inspect

from krauncher import KrauncherClient, KrauncherError
from krauncher.analyzer import AssayOutdated

client = KrauncherClient()


def train(epochs: int = 3, batch_size: int = 128):
    import torch
    import torchvision
    from torch.utils.data import DataLoader

    model = torchvision.models.resnet50(num_classes=10).cuda()
    data = torchvision.datasets.CIFAR10("data", train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(x.cuda()), y.cuda()).backward()
            opt.step()


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    # The analyzer this account is configured for (address and token come from
    # the broker). classify() returns the analyzer's job id with the result.
    analyzer = client.analyzer()
    c = await analyzer.classify(inspect.getsource(train), kwargs={"epochs": 3, "batch_size": 128})

    assay = await analyzer.assay(c.analyzer_job_id)
    work = assay["work"]
    print(f"Workload:       {assay['workload']['type']}, "
          f"min VRAM {assay['requirements']['min_vram_gb']} GB")
    print(f"Reference card: {work['reference_cu']} CU = {work['reference_sec']} s "
          f"(a slow host up to x{work['spread']['factor']})")

    try:
        ladder = await client.ladder(assay)
    except AssayOutdated as exc:
        print(f"\nThe analyzer was recalibrated after this assay: analyze the code again. {exc}")
        return
    except KrauncherError as exc:
        print(f"\n{exc}")
        return

    # Seconds per CU on the reference card: the assay's own scale.
    sec_per_cu = work["reference_sec"] / work["reference_cu"]
    compute_sec = work["phases_cu"]["compute"] * sec_per_cu
    other_sec = work["reference_sec"] - compute_sec

    print()
    print(f"{'GPU':<28}{'VRAM, GB':>9}{'compute x':>11}{'time, s':>10}")
    print("─" * 58)
    for row in ladder["rows"][:15]:
        sec = row["compute_ratio"] * compute_sec + other_sec
        print(f"{row['gpu_name']:<28}{row['vram_gb']:>9}"
              f"{row['compute_ratio']:>11.3f}{sec:>10.1f}")
    print(f"... {len(ladder['rows'])} GPUs, calibration {ladder['meta']['calibration_id']}")

if __name__ == "__main__":
    asyncio.run(main())
