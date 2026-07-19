"""Tutorial 23: Graph Convolutional Network (GCN) node classification on Cora.

Trains a 2-layer GCN (Kipf & Welling, 2017) on the Cora citation graph
to classify papers into 7 research topics. Canonical "hello world" of
Graph Neural Networks.

Why this tutorial: GNNs are massively popular (drug discovery,
recommender systems, fraud detection, social-graph analysis), but the
analyzer's AST classifier has no `task_pattern` entry for graph
workloads — arch_type stays None, no SUPPORTED_TASK_PATTERN matches,
and the request is routed to LLMFullPredictor for an anchored CU
estimate.

Dataset: Cora — 2708 nodes (papers), 5429 edges (citations), 1433-dim
bag-of-words features, 7 classes. Downloaded by torch_geometric (~5 MB).

Model: GCN, ~50K params (1433 → 16 → 7). Tiny by GPU standards but the
graph propagation pattern is structurally different from a CNN/MLP.

Expected runtime: ~10-20 s on RTX 4090 (200 epochs full-batch training).
"""

import asyncio

from krauncher import KrauncherClient

client = KrauncherClient()


@client.task(
    vram_gb=4,
    timeout=600,
    pip=["torch_geometric"],
    dataset_size=5,  # Cora ~5 MB, fetched by PyG into /tmp
    disk_gb=4,
)
def gcn_cora(
    num_epochs: int = 200,
    hidden_channels: int = 16,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
):
    """Train a 2-layer GCN on Cora for paper-topic classification."""
    print("Task started. Importing torch / torch_geometric (~5-10s)...",
          flush=True)
    import time

    _t_imp = time.monotonic()
    import torch
    import torch.nn.functional as F
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GCNConv
    print(f"Imports done in {time.monotonic() - _t_imp:.1f}s.", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading Cora dataset...", flush=True)
    dataset = Planetoid(root="/tmp/cora", name="Cora")
    data = dataset[0].to(device)
    print(f"  nodes={data.num_nodes}, edges={data.num_edges}, "
          f"features={dataset.num_node_features}, classes={dataset.num_classes}",
          flush=True)

    class GCN(torch.nn.Module):
        def __init__(self, in_channels: int, hidden: int, out_channels: int):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    model = GCN(
        in_channels=dataset.num_node_features,
        hidden=hidden_channels,
        out_channels=dataset.num_classes,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"GCN params: {params:,}", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    def train_step():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def eval_acc():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        accs = {}
        for split in ("train_mask", "val_mask", "test_mask"):
            mask = getattr(data, split)
            accs[split] = (pred[mask] == data.y[mask]).float().mean().item()
        return accs

    t0 = time.monotonic()
    best_val_acc = 0.0
    best_test_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        loss = train_step()
        if epoch % 20 == 0 or epoch == 1:
            accs = eval_acc()
            if accs["val_mask"] > best_val_acc:
                best_val_acc = accs["val_mask"]
                best_test_acc = accs["test_mask"]
            print(f"epoch {epoch:3d}/{num_epochs}  loss={loss:.4f}  "
                  f"train={accs['train_mask']:.3f}  "
                  f"val={accs['val_mask']:.3f}  "
                  f"test={accs['test_mask']:.3f}",
                  flush=True)

    total = time.monotonic() - t0
    final_accs = eval_acc()
    print(f"Training done in {total:.1f}s. "
          f"final test_acc={final_accs['test_mask']:.4f}, "
          f"best test_acc={best_test_acc:.4f}", flush=True)

    return {
        "num_epochs": num_epochs,
        "hidden_channels": hidden_channels,
        "num_params": params,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "final_test_acc": round(final_accs["test_mask"], 4),
        "best_test_acc": round(best_test_acc, 4),
        "training_sec": round(total, 2),
    }


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    print("Submitting GCN node classification on Cora...")
    print("  Model:    2-layer GCN, hidden=16, ~50K params")
    print("  Dataset:  Cora (2708 nodes, 5429 edges, 7 classes)")
    print("  Epochs:   200")
    print("  Expected: ~10-20 s on RTX 4090")
    handle = await gcn_cora()
    print(f"Task submitted: {handle.task_id}")

    def on_log(msg: dict):
        if msg.get("type") not in ("stdout", "stderr"):
            return
        text = (msg.get("data") or {}).get("text") or ""
        for line in text.splitlines():
            low = line.lower()
            if any(k in low for k in (
                "epoch ", "training done", "loading",
                "device:", "gcn params",
            )):
                print(f"  {line.rstrip()}")

    result = await handle.wait(on_log=on_log, timeout=900)

    output = result.output
    print("\nResults:")
    print(f"  Epochs:           {output['num_epochs']}")
    print(f"  GCN params:       {output['num_params']:,}")
    print(f"  Cora nodes/edges: {output['num_nodes']} / {output['num_edges']}")
    print(f"  Final test acc:   {output['final_test_acc']}")
    print(f"  Best test acc:    {output['best_test_acc']}")
    print(f"  Training time:    {output['training_sec']} s")


if __name__ == "__main__":
    asyncio.run(main())
