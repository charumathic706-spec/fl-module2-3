import numpy as np

from module1.split3.model_hasher import ModelHasher, simulate_tamper


def test_hash_chain_tamper_detection():
    hasher = ModelHasher()

    params_r1 = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]
    params_r2 = [np.array([[1.1, 2.1], [3.1, 4.1]], dtype=np.float32)]

    hasher.hash_round(1, params_r1)
    hasher.hash_round(2, params_r2)

    intact = hasher.verify_chain()
    assert intact.is_intact

    tampered = simulate_tamper(hasher, round_num=1)
    report = tampered.verify_chain()

    assert not report.is_intact
    assert 1 in report.tampered_rounds
