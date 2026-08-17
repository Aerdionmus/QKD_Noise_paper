from qkd_noise.metrics import binary_entropy, pri, secret_key_rate


def test_binary_entropy_half():
    assert abs(binary_entropy(0.5) - 1.0) < 1e-12


def test_key_rate_matches_equation_19():
    q = 0.05
    mu = 0.1
    delta = 0.0
    expected = max(0.0, 1 - 2 * binary_entropy(q)) * (1 - mu / 2) - delta
    assert abs(secret_key_rate(q, mu, delta) - expected) < 1e-12


def test_pri_definition():
    assert pri(0.6, 0.03) == 20.0
