import gymnasium as gym
import numpy as np

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
        
        for i in range(self.repeat):
            obs, reward, d, t, info = self.env.step(action)
            
            total_reward += reward
            done = d or done
            truncated = t or truncated
            
            # 嘗試累積單步手續費資訊 (如果存在)
            if 'step_fee_ratio' in info:
                total_step_fee += info['step_fee_ratio']
            
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
            
        return obs, total_reward, done, truncated, info


class ActionSmoothClipWrapper(gym.ActionWrapper):
    """
    先對動作做持倉硬上限裁剪，再做指數平滑，降低高頻翻倉與換手。
    
    max_position_pct: 目標持倉百分比上限（-P~P）
    smoothing_alpha: 平滑係數；越小越平滑，0.3~0.5 常用
    """
    def __init__(self, env, *, max_position_pct: float, smoothing_alpha: float = 0.3):
        super().__init__(env)
        assert max_position_pct > 0.0, "max_position_pct must be positive"
        assert 0.0 < smoothing_alpha <= 1.0, "smoothing_alpha must be in (0, 1]"
        self.max_position_pct = float(max_position_pct)
        self.smoothing_alpha = float(smoothing_alpha)
        # 為每個環境維持上一動作，用同型態/shape 初始化
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return super().reset(**kwargs)

    def action(self, action):
        # clip -> smooth，輸出保持 float32
        clipped = np.clip(action, -self.max_position_pct, self.max_position_pct)
        smoothed = (
            self.smoothing_alpha * clipped
            + (1.0 - self.smoothing_alpha) * self.prev_action
        )
        self.prev_action = smoothed.astype(np.float32)
        return self.prev_action