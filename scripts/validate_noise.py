from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qkd_noise.channels import bb84_average_qber
from qkd_noise.config import PaperConfig
from qkd_noise.metrics import secret_key_rate


def main():
    cfg = PaperConfig()

    print("=== Paper parameter snapshot ===")
    print(cfg)

    print("\n=== Exact BB84 QBER from stated Kraus channels ===")
    print("p      single        dual          triple")
    for p in (0.01, 0.02, 0.03, 0.05, 0.10):
        vals = [bb84_average_qber(p, s) for s in ("single", "dual", "triple")]
        print(f"{p:0.02f}   " + "   ".join(f"{v:0.6f}" for v in vals))

    print("\n=== Paper practical terms ===")
    print(f"Q_align = {cfg.q_alignment:.6f}")
    print(f"Q0 reported by paper = {cfg.q0_reported:.6f}")
    print(f"ΔPNS literal = {cfg.delta_pns:.6g}")
    print(f"Dark-count Eq.(13) literal = {cfg.dark_count_qber_paper_literal:.6g}")

    print("\n=== PRI sanity check from Eq.(20) ===")
    for scenario, sigma in (("single", 0.03), ("dual", 0.06), ("triple", 0.09)):
        print(
            f"{scenario:7s}: Σp={sigma:.2f}, "
            f"PRI upper bound if R<=1 = {1/sigma:.4f}"
        )

    print("\n=== Key-rate examples using Eq.(19) ===")
    for q in (0.05, 0.10, 0.11, 0.14):
        r = secret_key_rate(q, cfg.mean_photon_number, cfg.delta_pns)
        print(f"Q={q:.3f} -> R={r:.6f}")


if __name__ == "__main__":
    main()
