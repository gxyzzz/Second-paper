from pathlib import Path

import pandas as pd


def test_protocol_edge_key_invariant():
    fit = pd.DataFrame({"userID": [0, 0, 1], "itemID": [1, 2, 3]})
    probe = pd.DataFrame({"userID": [0, 1], "itemID": [4, 5]})
    assert not (set(map(tuple, fit.values)) & set(map(tuple, probe.values)))


def test_round1_config_keeps_confirm_and_test_closed():
    import yaml

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/round1_baby.yaml").read_text())
    assert cfg["access"]["confirm_open"] is False
    assert cfg["access"]["test_open"] is False
    assert cfg["window_start_rank"] == 6
    assert cfg["window_end_rank"] == 30
    assert cfg["residual"]["eta_candidates"] == [0.0, 0.05, 0.1, 0.2]

if __name__ == "__main__":
    test_protocol_edge_key_invariant()
    test_round1_config_keeps_confirm_and_test_closed()
    print("ROUND1_PROTOCOL_TESTS_PASS")
