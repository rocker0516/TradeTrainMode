import math

import numpy as np

from Env.reward import (
    RewardCalculator,
    RewardWeights,
    RewardNormalizer,
    RewardNormalizationConfig,
    RewardSignals,
)


def test_reward_calculator_basic_composition():
    weights = RewardWeights(
        pnl=0.4,
        entry=0.05,
        stop=0.05,
        holding=0.03,
        penalty=0.02,
        risk_management=0.45,
    )
    normalizer = RewardNormalizer(RewardNormalizationConfig())
    calc = RewardCalculator(weights=weights, normalizer=normalizer)

    signals = RewardSignals(
        pnl_ratio=0.02,      # 小幅獲利
        entry_score=0.001,   # 小幅正向
        stop_score=0.002,    # 正向事件（止盈/止損觸發）
        holding_score=0.001, # 小幅正向
        penalty_score=-0.001,# 負向懲罰
        risk_score=0.01,     # 小幅正向風險管理獎勵
    )

    reward = calc.calculate(signals)

    assert isinstance(reward, float)
    assert -1.0 <= reward <= 1.0


def test_penalty_asymmetry_positive_small_bonus():
    calc = RewardCalculator()
    signals = RewardSignals(penalty_score=0.001)
    reward = calc.calculate(signals)
    # 確保正向 penalty 的獎勵有封頂（上限）
    assert reward <= 0.1


def test_tanh_squashing_for_large_pnl():
    calc = RewardCalculator()
    signals = RewardSignals(pnl_ratio=1.0)
    reward = calc.calculate(signals)
    # tanh 會接近飽和於 1，但仍受權重與最終裁剪影響
    assert reward <= 1.0
    assert reward >= 0.0


