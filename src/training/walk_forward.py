import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models.hstgt import HSTGT
from .loss import RiskAwareLoss, directional_accuracy


def load_graph(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_fold(
    model: HSTGT,
    graph_paths: list[str],
    loss_fn: RiskAwareLoss,
    device: str,
) -> dict:
    """Evaluate model on a list of daily graphs. Returns metric dict."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0

    with torch.no_grad():
        for path in graph_paths:
            try:
                data = load_graph(path)
                data = data.to(device)
                preds = model(data)
                y     = data["stock"].y

                loss, _ = loss_fn(preds, y)
                total_loss += loss.item()
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
            except Exception as e:
                continue

    if not all_preds:
        return {}

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    da = directional_accuracy(preds, targets)

    # Simulated daily portfolio return: long top quartile, short bottom quartile (strategy btw)
    portfolio_returns = []
    for p_day, t_day in zip(all_preds, all_targets):
        if len(p_day) < 4:
            continue
        q75 = torch.quantile(p_day, 0.75)
        q25 = torch.quantile(p_day, 0.25)
        long_mask  = p_day >= q75
        short_mask = p_day <= q25
        if long_mask.sum() == 0 or short_mask.sum() == 0:
            continue
        daily_ret = t_day[long_mask].mean() - t_day[short_mask].mean()
        portfolio_returns.append(daily_ret.item())

    if len(portfolio_returns) > 1:
        ret_array = np.array(portfolio_returns)
        sharpe = (ret_array.mean() / (ret_array.std() + 1e-8)) * np.sqrt(252)
    else:
        sharpe = 0.0

    rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()

    return {
        "loss":  total_loss / max(len(graph_paths), 1),
        "rmse":  rmse,
        "da":    da,
        "sharpe": sharpe,
    }


def walk_forward_train(
    graph_paths: list[str],
    model_config: dict,
    train_days:  int = 750,
    val_days:    int = 60,
    test_days:   int = 30,
    stride:      int = 30,
    lr:          float = 1e-4,
    epochs:      int = 20,
    patience:    int = 5,
    device:      str = "auto",
    output_dir:  str = "results",
    loss_kwargs: dict = None,
) -> pd.DataFrame:
    """
    Rolling walk-forward loop.

    Timeline for one fold:
    |-------- train_days --------|-- val_days --|-- test_days --|
    Then slide the entire window forward by `stride` days.

    Returns a DataFrame of per-fold test metrics.
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"Training on: {device}")

    os.makedirs(output_dir, exist_ok=True)
    loss_kwargs = loss_kwargs or {"lambda_mse": 1.0, "lambda_dir": 1.0}
    loss_fn     = RiskAwareLoss(**loss_kwargs).to(device)

    total_days  = len(graph_paths)
    window_size = train_days + val_days + test_days
    fold_results = []

    fold = 0
    start_idx = 0

    while start_idx + window_size <= total_days:
        fold += 1
        train_paths = graph_paths[start_idx : start_idx + train_days]
        val_paths   = graph_paths[start_idx + train_days : start_idx + train_days + val_days]
        test_paths  = graph_paths[start_idx + train_days + val_days : start_idx + window_size]

        print(f"\n{'='*60}")
        print(f"FOLD {fold} | Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
        print(f"{'='*60}")

        # Skip fold if checkpoint already exists
        ckpt_path = os.path.join(output_dir, f"fold_{fold}_model.pt")
        if os.path.exists(ckpt_path):
            print(f"  ⏭ Fold {fold} already done, skipping...")
            start_idx += stride
            continue

        # Load one graph to get metadata
        sample_data = load_graph(train_paths[0])
        metadata = sample_data.metadata()

        # Fresh model per fold (strict walk-forward)
        model = HSTGT(metadata=metadata, **model_config).to(device)
        optimiser = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimiser, T_max=epochs)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        # ── Training loop ──────────────────────────────────────────────────
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            np.random.shuffle(train_paths)  # Shuffle within training window

            for path in train_paths:
                try:
                    data = load_graph(path)
                    data = data.to(device)

                    optimiser.zero_grad()
                    preds = model(data)
                    y     = data["stock"].y

                    loss, components = loss_fn(preds, y)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimiser.step()
                    epoch_loss += loss.item()
                except Exception:
                    continue

            scheduler.step()

            # ── Validation ─────────────────────────────────────────────────
            val_metrics = evaluate_fold(model, val_paths, loss_fn, device)
            val_loss    = val_metrics.get("loss", float("inf"))

            print(
                f"  Epoch {epoch:3d} | "
                f"Train Loss: {epoch_loss/max(len(train_paths),1):.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val DA: {val_metrics.get('da', 0):.2%} | "
                f"Val Sharpe: {val_metrics.get('sharpe', 0):.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  ⏹ Early stopping at epoch {epoch}")
                    break

        # ── Test evaluation ────────────────────────────────────────────────
        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        test_metrics = evaluate_fold(model, test_paths, loss_fn, device)
        print(
            f"\n  ★ FOLD {fold} TEST | "
            f"RMSE: {test_metrics.get('rmse', 0):.4f} | "
            f"DA: {test_metrics.get('da', 0):.2%} | "
            f"Sharpe: {test_metrics.get('sharpe', 0):.2f}"
        )

        fold_results.append({"fold": fold, **test_metrics})

        # Save model checkpoint
        ckpt_path = os.path.join(output_dir, f"fold_{fold}_model.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint → {ckpt_path}")

        start_idx += stride

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(output_dir, "walk_forward_results.csv"), index=False)

    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL FOLDS:")
    print(results_df.describe().round(4))
    return results_df