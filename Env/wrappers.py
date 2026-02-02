import gymnasium as gym
import numpy as np
from collections import defaultdict

class ActionRepeatWrapper(gym.Wrapper):
    """
    Action Repeat Wrapper (Frame Skipping)
    
    重複執行相同的動作 N 次，並累加獎勵。
    這可以減少 Agent 的決策頻率，強迫其關注更長期的結果，
    並自然降低交易頻率與換手率。

    Safety Features:
    - 若在 repeat 期間觸發止損 (Stop Loss) 或 強平 (Liquidation)，立即中止循環，
      讓 Agent 能在下一步重新決策（例如反向開倉或觀望）。
    """
    def __init__(self, env, repeat: int = 1):
        super().__init__(env)
        assert repeat >= 1, "Repeat count must be at least 1"
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = {}
        
        # 累積變數
        total_step_fee = 0.0
        total_cost = 0.0
        total_idle_penalty = 0.0
        # 注意：訓練端的 LagrangianCallback 會從 info 讀取 cost_* 來更新 λ，
        # 若 repeat>1 但只保留「最後一步」的 cost_*，會導致 avg_cost 低估甚至顯示為 0，造成你以為「違規卻不更新」。
        total_cost_channels = defaultdict(float)   # e.g. cost_risk / cost_fric / cost_sl_buf / cost_sl_event
        total_cost_breakdown = defaultdict(float)  # e.g. death_cost / fric_cost / sl_buf_cost / stop_missing_cost
        
        for i in range(self.repeat):
            obs, reward, d, t, info = self.env.step(action)
            
            total_reward += reward
            done = d or done
            truncated = t or truncated
            
            # 嘗試累積單步手續費資訊 (如果存在)
            if 'step_fee_ratio' in info:
                total_step_fee += info['step_fee_ratio']
            # 嘗試累積 cost（如果存在）
            if 'cost' in info:
                total_cost += float(info['cost'])
            # 嘗試累積多通道 cost（如果存在）
            for k in ("cost_risk", "cost_fric", "cost_sl_buf", "cost_sl_event"):
                if k in info:
                    try:
                        total_cost_channels[k] += float(info.get(k, 0.0))
                    except (TypeError, ValueError):
                        pass
            # 嘗試累積 cost_breakdown（如果存在）
            breakdown = info.get("cost_breakdown")
            if isinstance(breakdown, dict):
                for bk, bv in breakdown.items():
                    try:
                        total_cost_breakdown[str(bk)] += float(bv)
                    except (TypeError, ValueError):
                        pass
            if "idle_penalty" in info:
                total_idle_penalty += float(info.get("idle_penalty", 0.0))
            
            # --- Safety Break Logic ---
            # 如果觸發止損、強平或任何終止條件，立即停止 Repeat，
            # 把控制權交還給 Agent。
            stop_loss_triggered = info.get('stop_loss_triggered', False)
            liq_triggered = info.get('liq_triggered', False)
            
            if done or truncated or stop_loss_triggered or liq_triggered:
                break
        
        # 更新 Info 中的累積值 (僅針對需要加總的欄位)
        if 'step_fee_ratio' in info:
            info['step_fee_ratio'] = total_step_fee
        if 'cost' in info:
            info['cost'] = float(total_cost)
        # 同步回填 multi-channel costs（供 Lagrangian / 觀測 / 日誌使用）
        for k, v in total_cost_channels.items():
            info[k] = float(v)
        # 同步回填 breakdown（避免 cost 與 breakdown 量級不一致）
        if total_cost_breakdown:
            info["cost_breakdown"] = dict(total_cost_breakdown)
        info["idle_penalty"] = float(total_idle_penalty)
            
        return obs, total_reward, done, truncated, info


class ActionClipWrapper(gym.ActionWrapper):
    """
    動作截斷（Clip）Wrapper

    用途：
    - 限制 Agent 的「目標持倉百分比」上限（-P ~ P），避免輸出極端動作造成手續費暴增或不穩定。

    注意：
    - 本 Wrapper **不做 Action smoothing**（已移除 EMA 平滑邏輯）。
    
    max_position_pct: 目標持倉百分比上限（-P~P）
    """
    def __init__(self, env, *, max_position_pct: float):
        super().__init__(env)
        assert max_position_pct > 0.0, "max_position_pct must be positive"
        self.max_position_pct = float(max_position_pct)

    def action(self, action):
        # clip，輸出保持 float32
        clipped = np.clip(action, -self.max_position_pct, self.max_position_pct)
        return clipped.astype(np.float32)