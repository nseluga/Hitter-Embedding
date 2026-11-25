"""One check: seen_rows drops cold-start (index 0) hitters and keeps order."""
import torch
from src.analysis.final_loss_report import seen_rows


def test_seen_rows_drops_cold_start():
    hitter = torch.tensor([0, 3, 0, 5, 7])
    assert seen_rows(hitter, torch.tensor([0, 1, 2, 3, 4])).tolist() == [1, 3, 4]
    assert seen_rows(hitter, torch.tensor([3, 2])).tolist() == [3]
