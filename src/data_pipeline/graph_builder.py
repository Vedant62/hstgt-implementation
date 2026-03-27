import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm
from typing import Optional


def build_correlation_edges(
    returns_window: pd.DataFrame,      # shape: (window_days, n_stocks)
    threshold: float = 0.6,
    tickers: list[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    corr_matrix = returns_window.corr().fillna(0).values  # (N, N)
    n = corr_matrix.shape[0]

    src_list, dst_list, attr_list = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = corr_matrix[i, j]
            if abs(c) >= threshold:
                src_list.append(i)
                dst_list.append(j)
                attr_list.append([c])  # signed correlation as weight

    if not src_list:
        # Ensure at least empty tensors with correct shape
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list, dtype=torch.float)
    return edge_index, edge_attr


def build_sector_edges(
    tickers: list[str],
    gics_df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor]:

    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    gics_filtered = gics_df[gics_df["ticker"].isin(tickers)].copy()

    src_list, dst_list = [], []
    for _, group in gics_filtered.groupby("sub_industry"):
        group_tickers = group["ticker"].tolist()
        for i, ta in enumerate(group_tickers):
            for tb in group_tickers[i+1:]:
                if ta in ticker_to_idx and tb in ticker_to_idx:
                    ia, ib = ticker_to_idx[ta], ticker_to_idx[tb]
                    src_list += [ia, ib]   # bidirectional
                    dst_list += [ib, ia]

    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.ones((edge_index.shape[1], 1), dtype=torch.float)
    return edge_index, edge_attr

def build_semantic_edges(
    tickers: list[str],
    semantic_scores_path: Optional[str] = None,
    date_str: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    if semantic_scores_path is None or not os.path.exists(semantic_scores_path):
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))

    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    scores = pd.read_parquet(semantic_scores_path)
    if date_str:
        scores = scores[scores["date"] == date_str]

    src_list, dst_list, attr_list = [], [], []
    for _, row in scores.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        score  = float(row["spillover_score"])
        if score < 0.3:  # Threshold: only meaningful connections
            continue
        if ta in ticker_to_idx and tb in ticker_to_idx:
            ia, ib = ticker_to_idx[ta], ticker_to_idx[tb]
            src_list += [ia, ib]
            dst_list += [ib, ia]
            attr_list += [[score], [score]]

    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list, dtype=torch.float)
    return edge_index, edge_attr


# ── Main graph builder ─────────────────────────────────────────────────────────

def build_daily_hetero_graph(
    date: str,
    tickers: list[str],
    feature_panel: dict[str, pd.DataFrame], #AAPL -> (30, 8)
    gics_df: pd.DataFrame,
    corr_window: int = 30,
    corr_threshold: float = 0.6,
    semantic_scores_path: Optional[str] = None,
    seq_len: int = 30,
) -> Optional[HeteroData]:
    """
    Node type: 'stock'
    Edge types:
        ('stock', 'mathematical', 'stock')  <- rolling correlation
        ('stock', 'structural', 'stock')    <- GICS sector
        ('stock', 'semantic', 'stock')      <- news co-mention
    """
    date = pd.Timestamp(date)

    node_feats, targets, valid_tickers = [], [], []
    feature_cols = [
        "log_return", "volatility_30d", "rsi_14",
        "macd", "macd_signal", "bb_upper", "bb_lower", "volume"
    ]

    returns_window_dict = {}  

    for ticker in tickers:
        if ticker not in feature_panel:
            continue
        df = feature_panel[ticker]
        hist = df[df.index <= date].tail(seq_len + corr_window)

        if len(hist) < seq_len + 1:
            continue  

        seq = hist[feature_cols].iloc[-seq_len:].values  # (seq_len, F)

        target_row = hist.iloc[-1]
        if pd.isna(target_row["target_return"]):
            continue

        node_feats.append(seq)
        targets.append(target_row["target_return"])
        valid_tickers.append(ticker)

        # For correlation: just use log returns of the corr_window
        returns_window_dict[ticker] = hist["log_return"].iloc[-(seq_len + corr_window):-seq_len]

    if len(valid_tickers) < 5:
        return None  # Too few stocks to build a meaningful graph

    x = torch.tensor(np.stack(node_feats), dtype=torch.float)  # (N, T, F)
    y = torch.tensor(targets, dtype=torch.float)                # (N,)

    returns_window = pd.DataFrame(returns_window_dict).fillna(0)

    data = HeteroData()
    data["stock"].x = x
    data["stock"].y = y
    data["stock"].tickers = valid_tickers

    ei_math, ea_math = build_correlation_edges(returns_window, corr_threshold, valid_tickers)
    data["stock", "mathematical", "stock"].edge_index = ei_math
    data["stock", "mathematical", "stock"].edge_attr  = ea_math

    ei_struct, ea_struct = build_sector_edges(valid_tickers, gics_df)
    data["stock", "structural", "stock"].edge_index = ei_struct
    data["stock", "structural", "stock"].edge_attr  = ea_struct

    ei_sem, ea_sem = build_semantic_edges(
        valid_tickers, semantic_scores_path, str(date.date())
    )
    data["stock", "semantic", "stock"].edge_index = ei_sem
    data["stock", "semantic", "stock"].edge_attr  = ea_sem

    return data


def build_graph_dataset(
    trading_dates: list[str],
    tickers: list[str],
    feature_panel: dict[str, pd.DataFrame],
    gics_df: pd.DataFrame,
    output_dir: str = "data/graphs",
    **kwargs,
) -> list[str]:
    """
    Build and cache all daily graphs. Returns list of cached file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for date_str in tqdm(trading_dates, desc="Building Graphs"):
        out_path = os.path.join(output_dir, f"{date_str}.pkl")
        if os.path.exists(out_path):
            paths.append(out_path)
            continue

        graph = build_daily_hetero_graph(
            date=date_str,
            tickers=tickers,
            feature_panel=feature_panel,
            gics_df=gics_df,
            **kwargs,
        )
        if graph is not None:
            with open(out_path, "wb") as f:
                pickle.dump(graph, f)
            paths.append(out_path)

    print(f"Built {len(paths)} daily graphs → {output_dir}")
    return paths