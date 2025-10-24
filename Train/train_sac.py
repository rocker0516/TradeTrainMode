"""
SAC 交易模型訓練腳本（使用 Stable-Baselines3）

使用 SB3 內建的 SAC 算法進行訓練。
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加父目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from gymnasium.wrappers import TimeLimit

from Env.trading_env import TradingEnvironment
from Env.reward import RewardCalculator


class TradingEnvFactory:
    """
    可序列化的環境工廠，用於多進程向量環境。

    每個子進程都會持有各自獨立的 RewardCalculator 與環境狀態，
    以確保「每回合獨立獎勵計算」。
    """

    def __init__(
        self,
        *,
        data_path: str,
        initial_balance: float,
        transaction_fee: float,
        window_size: int,
        leverage: float,
        min_balance: float,
        min_trade_qty: float,
        margin_mode: str,
        reward_mode: str,
        random_start: bool,
    ) -> None:
        self.data_path = data_path
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.leverage = leverage
        self.min_balance = min_balance
        self.min_trade_qty = min_trade_qty
        self.margin_mode = margin_mode
        self.reward_mode = reward_mode
        self.random_start = random_start

    def __call__(self):
        df = pd.read_csv(self.data_path)
        rc = RewardCalculator(mode=self.reward_mode, scale=1.0)
        env = TradingEnvironment(
            df=df,
            initial_balance=self.initial_balance,
            transaction_fee=self.transaction_fee,
            window_size=self.window_size,
            leverage=self.leverage,
            min_balance=self.min_balance,
            min_trade_qty=self.min_trade_qty,
            margin_mode=self.margin_mode,
            reward_calculator=rc,
            random_start=self.random_start,
        )
        return Monitor(env)


class TradingCallback(BaseCallback):
    """
    自定義訓練回調
    
    記錄訓練過程中的詳細信息。
    """
    
    def __init__(self, log_interval: int = 1, verbose: int = 1, print_step_episode: bool = False):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.print_step_episode = bool(print_step_episode)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_profit_rates = []
        self.episode_failures = []  # True 表示失敗（非 data_exhausted）
        # 向量環境的逐環境狀態
        self.current_episode_reward = None  # 將在 training start 時初始化為 (n_envs,) 向量
        self.current_episode_length = None
        self.env_episode_indices = None  # 每個環境的回合序號
    
    def _on_training_start(self) -> None:
        try:
            n_envs = getattr(self.training_env, 'num_envs', 1)
        except Exception:
            n_envs = 1
        self.current_episode_reward = np.zeros(n_envs, dtype=float)
        self.current_episode_length = np.zeros(n_envs, dtype=int)
        self.env_episode_indices = np.ones(n_envs, dtype=int)

    def _on_step(self) -> bool:
        """每步後調用（支援向量環境）"""
        rewards = self.locals.get('rewards')
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        if rewards is None or dones is None:
            return True

        n_envs = len(rewards)
        # 累積獎勵與步數
        self.current_episode_reward[:n_envs] += rewards
        self.current_episode_length[:n_envs] += 1

        # 可選：每步輸出任一環境的回合數
        if self.print_step_episode:
            try:
                print(f"[train] step={self.num_timesteps}", end='\r')
            except Exception:
                pass

        # 處理結束的環境們
        for i in range(n_envs):
            if not dones[i]:
                continue

            # 取得該環境的底層 env
            try:
                wrapped_env = self.training_env.envs[i]
                env = wrapped_env.unwrapped if hasattr(wrapped_env, 'unwrapped') else wrapped_env
            except Exception:
                env = None

            final_balance = None
            profit_rate = None
            term_result = 'unknown'
            failure_flag = False
            long_close_count = None
            short_close_count = None
            total_fees = None
            episode_steps = None
            long_entry_count = None
            short_entry_count = None
            episode_max_steps = None

            # 從 infos[i] 讀取結算資訊
            try:
                info_i = infos[i] if isinstance(infos, (list, tuple)) and i < len(infos) else {}
                if isinstance(info_i, dict):
                    reason = info_i.get('termination_reason')
                    failure_flag = (reason is not None and reason != 'data_exhausted')
                    if 'final_balance' in info_i:
                        final_balance = float(info_i['final_balance'])
                        profit_rate = float(info_i.get('profit_rate', 0.0))
                    if 'long_close_count' in info_i:
                        long_close_count = int(info_i['long_close_count'])
                    if 'short_close_count' in info_i:
                        short_close_count = int(info_i['short_close_count'])
                    if 'total_fees' in info_i:
                        total_fees = float(info_i['total_fees'])
                    if 'episode_steps' in info_i:
                        episode_steps = int(info_i['episode_steps'])
                    if 'long_entry_count' in info_i:
                        long_entry_count = int(info_i['long_entry_count'])
                    if 'short_entry_count' in info_i:
                        short_entry_count = int(info_i['short_entry_count'])
                    if 'episode_max_steps' in info_i:
                        episode_max_steps = int(info_i['episode_max_steps'])
                    if reason == 'data_exhausted':
                        term_result = 'success(data_exhausted)'
                    elif reason is not None:
                        term_result = f"fail({reason})"
            except Exception:
                pass

            # 若 info 未提供，則從 env 讀取
            if (final_balance is None or profit_rate is None) and env is not None:
                try:
                    final_balance = float(env.total_value)
                    profit = final_balance - env.initial_balance
                    profit_rate = (profit / env.initial_balance) * 100
                except Exception:
                    final_balance = 0.0
                    profit_rate = 0.0

            # 記錄聚合回合資料
            self.episode_rewards.append(self.current_episode_reward[i])
            self.episode_lengths.append(int(self.current_episode_length[i]))
            self.episode_profit_rates.append(float(profit_rate))
            self.episode_failures.append(bool(failure_flag))

            episode_num = len(self.episode_rewards)
            if episode_num % self.log_interval == 0:
                print(
                    f"Episode {episode_num:4d} | "
                    f"Env#{i} | "
                    f"Steps: {episode_steps if episode_steps is not None else int(self.current_episode_length[i]):4d}/{episode_max_steps if episode_max_steps is not None else '?'} | "
                    f"Reward: {self.current_episode_reward[i]:8.2f} | "
                    f"Balance: {final_balance if final_balance is not None else 0.0:10.2f} | "
                    f"ProfitRate: {profit_rate if profit_rate is not None else 0.0:+.2f}% | "
                    f"Fees: {total_fees if total_fees is not None else 0.0:.4f} | "
                    f"Entries L/S: {long_entry_count if long_entry_count is not None else 0}/{short_entry_count if short_entry_count is not None else 0} | "
                    f"Result: {term_result}"
                )

            # 每 10 回合統計
            if episode_num % 10 == 0:
                last10_rates = self.episode_profit_rates[-10:]
                last10_fails = self.episode_failures[-10:]
                avg_rate_10 = float(np.mean(last10_rates)) if len(last10_rates) > 0 else 0.0
                total_fail_10 = int(np.sum(last10_fails)) if len(last10_fails) > 0 else 0
                print(
                    f"[10-episode stats] avg_profit_rate={avg_rate_10:+.2f}% | failures={total_fail_10}/10"
                )

            # 重置該環境的回合累積器（獨立回合）
            self.current_episode_reward[i] = 0.0
            self.current_episode_length[i] = 0
            self.env_episode_indices[i] += 1

        return True
    
    def _on_training_end(self) -> None:
        """訓練結束時調用"""
        if len(self.episode_rewards) > 0:
            print(f"\n{'='*60}")
            print("訓練總結:")
            print(f"  總回合數: {len(self.episode_rewards)}")
            print(f"  平均獎勵: {np.mean(self.episode_rewards):.2f}")
            print(f"  最佳獎勵: {np.max(self.episode_rewards):.2f}")
            print(f"  最差獎勵: {np.min(self.episode_rewards):.2f}")
            if len(self.episode_profit_rates) > 0:
                print(f"  平均收益率: {np.mean(self.episode_profit_rates):+.2f}%")
            if len(self.episode_failures) > 0:
                print(f"  總失敗次數: {int(np.sum(self.episode_failures))}")
            print(f"{'='*60}\n")


class VerboseEvalCallback(EvalCallback):
    """
    評估時打印提示訊息的回調。
    每當達到 eval_freq 觸發評估時，會打印當前全域步數與設定。
    """
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq) == 0:
            try:
                print(f"\n----- 評估開始 (global_steps={self.num_timesteps}, eval_freq={self.eval_freq}, n_eval_episodes={self.n_eval_episodes}) -----")
            except Exception:
                print("\n----- 評估開始 -----")
        return super()._on_step()


class RestartingEvalCallback(BaseCallback):
    """
    自訂評估回調：固定頻率觸發評估，遇到非「數據用完」的終止會自動從頭重啟，並統計成功/失敗次數。
    成功: info['termination_reason'] == 'data_exhausted'
    失敗: 其他 terminated 或被 TimeLimit 截斷（truncated=True）
    """
    def __init__(self, eval_env, eval_freq: int = 10000, n_eval_episodes: int = 5, deterministic: bool = True, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = bool(deterministic)
        self.success_episodes = 0
        self.failure_episodes = 0
        self.truncated_episodes = 0

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq) == 0:
            print(f"\n----- 自訂評估開始 (global_steps={self.num_timesteps}, eval_freq={self.eval_freq}, n_eval_episodes={self.n_eval_episodes}) -----")
            self._run_evaluation()
        return True

    def _run_evaluation(self) -> None:
        success = 0
        failure = 0
        truncated_cnt = 0
        episodes = 0
        try:
            while episodes < self.n_eval_episodes:
                obs, _ = self.eval_env.reset()
                while True:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    if terminated or truncated:
                        if truncated:
                            truncated_cnt += 1
                            failure += 1
                        else:
                            reason = info.get('termination_reason') if isinstance(info, dict) else None
                            if reason == 'data_exhausted':
                                success += 1
                            else:
                                failure += 1
                        episodes += 1
                        break
        except Exception:
            print("\n[錯誤位置] 評估 (evaluation) 發生例外，即將拋出")
            raise
        self.success_episodes += success
        self.failure_episodes += failure
        self.truncated_episodes += truncated_cnt
        print(f"自訂評估結果: 本輪 success={success}, failure={failure}, truncated={truncated_cnt} | 累計 success={self.success_episodes}, failure={self.failure_episodes}")

class TerminationAwareEvalCallback(EvalCallback):
    """
    自訂評估：
    - 若 done 且 info['termination_reason'] != 'data_exhausted'，視為失敗，重啟一個新 episode，累計失敗計數
    - 若 done 且為 'data_exhausted'，視為成功，累計成功計數
    - 可與 TimeLimit 搭配使用，步數達上限時 SB3 會設 truncated；此處以 terminated/done 為主
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.success_episodes = 0
        self.failure_episodes = 0

    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_event(self) -> None:
        # 在單輪 evaluate_policy 完成後被調用，累計成功/失敗次數
        try:
            # 最近一次評估的 episode info 存在 self.last_eval_ep_info (SB3 內部不提供)
            # 這裡退一步：根據評估環境的最後 done 時印出的訊息做統計
            pass
        except Exception:
            pass


def create_environment(
    data_path: str,
    initial_balance: float = 10000.0,
    min_balance: float = 100.0,
    leverage: float = 10.0,
    transaction_fee: float = 0.001,
    window_size: int = 288,
    reward_mode: str = 'delta_equity',
    margin_mode: str = 'isolated',
    start_date: str | None = None,
    end_date: str | None = None,
    random_start: bool = False
) -> TradingEnvironment:
    """
    創建交易環境
    
    Args:
        data_path: 數據路徑
        initial_balance: 初始資金
        leverage: 槓桿倍數
        transaction_fee: 交易手續費率
        window_size: 觀察窗口大小
        reward_mode: 獎勵模式
        margin_mode: 保證金模式
        
    Returns:
        交易環境實例
    """
    # 加載數據
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"數據文件不存在: {data_path}")
    
    print(f"加載數據: {data_path}")
    df = pd.read_csv(data_path)

    # 可選：依日期範圍過濾
    if start_date is not None or end_date is not None:
        dt_col = None
        for candidate in ('datetime', 'date', 'time', 'timestamp'):
            if candidate in df.columns:
                dt_col = candidate
                break
        if dt_col is None:
            print("警告: 指定了日期範圍，但資料無日期欄位，跳過日期過濾")
        else:
            df[dt_col] = pd.to_datetime(df[dt_col])
            if start_date is not None:
                df = df[df[dt_col] >= pd.to_datetime(start_date)]
            if end_date is not None:
                df = df[df[dt_col] <= pd.to_datetime(end_date)]
            df = df.reset_index(drop=True)
            print(f"已按照日期範圍過濾: start={start_date}, end={end_date}, 形狀: {df.shape}")

    print(f"數據形狀: {df.shape}")
    
    # 創建獎勵計算器
    reward_calculator = RewardCalculator(
        mode=reward_mode,
        scale=1.0
    )
    
    # 創建環境
    env = TradingEnvironment(
        df=df,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee,
        window_size=window_size,
        leverage=leverage,
        min_balance=min_balance,
        min_trade_qty=0.001,
        margin_mode=margin_mode,
        reward_calculator=reward_calculator,
        random_start=random_start
    )
    
    print(f"環境創建成功:")
    print(f"  觀察空間: {env.observation_space.shape}")
    print(f"  最小資金: {min_balance}")
    print(f"  動作空間: {env.action_space.shape}")
    print(f"  初始資金: {initial_balance}")
    print(f"  槓桿倍數: {leverage}")
    print(f"  保證金模式: {margin_mode}")
    
    return env


def train_sac(
    env: TradingEnvironment,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    buffer_size: int = 100000,
    learning_starts: int = 1000,
    batch_size: int = 256,
    tau: float = 0.005,
    gamma: float = 0.995,
    model_dir: str = './models',
    log_dir: str = './logs',
    save_freq: int = 10000,
    eval_freq: int = 10000,
    eval_episodes: int = 100,
    device: str = 'auto',
    load_model: str | None = None,
    n_envs: int = 1,
    train_data_path: str | None = None,
    eval_data_path: str | None = None,
    eval_start_date: str | None = None,
    eval_end_date: str | None = None,
    eval_max_steps: int | None = None
) -> SAC:
    """
    訓練 SAC 模型
    
    Args:
        env: 訓練環境
        total_timesteps: 總訓練步數
        learning_rate: 學習率
        buffer_size: 經驗回放緩衝區大小
        learning_starts: 開始學習前的隨機步數
        batch_size: 批次大小
        tau: 軟更新係數
        gamma: 折扣因子
        model_dir: 模型保存目錄
        log_dir: 日誌目錄
        save_freq: 保存頻率
        eval_freq: 評估頻率
        eval_episodes: 評估回合數
        device: 訓練設備
        load_model: 加載已有模型路徑
        
    Returns:
        訓練好的 SAC 模型
    """
    # 創建目錄
    model_dir = Path(model_dir)
    log_dir = Path(log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 檢查設備
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用設備: {device}")
    
    # 建立向量化或單環境
    monitor_dir_path = log_dir / 'monitor'
    monitor_dir_path.mkdir(parents=True, exist_ok=True)

    if n_envs is not None and int(n_envs) > 1:
        print(f"建立向量化環境: n_envs={int(n_envs)} (SubprocVecEnv)")
        factory = TradingEnvFactory(
            data_path=(train_data_path or eval_data_path or './Data/BTCUSDT_futures_volume_5years_5min.csv'),
            initial_balance=env.initial_balance,
            transaction_fee=env.transaction_fee,
            window_size=env.window_size,
            leverage=env.leverage,
            min_balance=env.min_balance,
            min_trade_qty=env.min_trade_qty,
            margin_mode=env.margin_mode,
            reward_mode=(getattr(env.reward_calculator, 'mode', 'delta_equity') if hasattr(env, 'reward_calculator') else 'delta_equity'),
            random_start=True,
        )
        vec_env = make_vec_env(factory, n_envs=int(n_envs), vec_env_cls=SubprocVecEnv, monitor_dir=str(monitor_dir_path))
        env_for_training = vec_env
    else:
        env_for_training = Monitor(env)
    
    # 創建或加載模型
    if load_model and Path(load_model).exists():
        print(f"加載模型: {load_model}")
        model = SAC.load(load_model, env=env_for_training, device=device)
        print("模型加載成功")
    else:
        print("創建新模型...")
        model = SAC(
            policy="MlpPolicy",
            env=env_for_training,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            verbose=1,
            device=device,
            tensorboard_log=str(log_dir)
        )
        print("模型創建成功")
    
    # 創建回調
    callbacks = []
    
    # 訓練日誌回調
    training_callback = TradingCallback(log_interval=1, print_step_episode=True)
    callbacks.append(training_callback)
    
    # 檢查點回調
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(model_dir),
        name_prefix='sac_trading'
    )
    callbacks.append(checkpoint_callback)
    
    # 評估回調：若提供 eval_start/end_date 則使用指定切片，否則複製訓練切片
    if eval_freq and eval_freq > 0:
        if eval_start_date is not None or eval_end_date is not None or eval_data_path is not None:
            eval_env = create_environment(
                data_path=eval_data_path or './Data/BTCUSDT_futures_volume_5years_5min.csv',
                initial_balance=(env_for_training.get_attr('initial_balance')[0] if hasattr(env_for_training, 'get_attr') else env.unwrapped.initial_balance),
                leverage=(env_for_training.get_attr('leverage')[0] if hasattr(env_for_training, 'get_attr') else env.unwrapped.leverage),
                transaction_fee=(env_for_training.get_attr('transaction_fee')[0] if hasattr(env_for_training, 'get_attr') else env.unwrapped.transaction_fee),
                window_size=(env_for_training.get_attr('window_size')[0] if hasattr(env_for_training, 'get_attr') else env.unwrapped.window_size),
                reward_mode=(env_for_training.get_attr('reward_calculator')[0].mode if hasattr(env_for_training, 'get_attr') else (env.unwrapped.reward_calculator.mode if hasattr(env.unwrapped, 'reward_calculator') else 'delta_equity')),
                margin_mode=(env_for_training.get_attr('margin_mode')[0] if hasattr(env_for_training, 'get_attr') else env.unwrapped.margin_mode),
                start_date=eval_start_date,
                end_date=eval_end_date,
                random_start=False,
            )
            if eval_max_steps is not None and eval_max_steps > 0:
                eval_env = TimeLimit(eval_env, max_episode_steps=int(eval_max_steps))
            eval_env = Monitor(eval_env)
        else:
            if hasattr(env_for_training, 'get_attr'):
                initial_balance = env_for_training.get_attr('initial_balance')[0]
                transaction_fee = env_for_training.get_attr('transaction_fee')[0]
                window_size = env_for_training.get_attr('window_size')[0]
                leverage = env_for_training.get_attr('leverage')[0]
                min_balance = env_for_training.get_attr('min_balance')[0]
                min_trade_qty = env_for_training.get_attr('min_trade_qty')[0]
                margin_mode = env_for_training.get_attr('margin_mode')[0]
                reward_mode = env_for_training.get_attr('reward_calculator')[0].mode
                eval_env = create_environment(
                    data_path=eval_data_path or './Data/BTCUSDT_futures_volume_5years_5min.csv',
                    initial_balance=initial_balance,
                    leverage=leverage,
                    transaction_fee=transaction_fee,
                    window_size=window_size,
                    reward_mode=reward_mode,
                    margin_mode=margin_mode,
                    random_start=False,
                )
            else:
                base_env = env.unwrapped
                eval_env = TradingEnvironment(
                    df=base_env.df.copy(),
                    initial_balance=base_env.initial_balance,
                    transaction_fee=base_env.transaction_fee,
                    window_size=base_env.window_size,
                    leverage=base_env.leverage,
                    min_balance=base_env.min_balance,
                    min_trade_qty=base_env.min_trade_qty,
                    margin_mode=base_env.margin_mode,
                    reward_calculator=RewardCalculator(
                        mode=getattr(base_env.reward_calculator, 'mode', 'delta_equity'),
                        scale=1.0
                    ),
                    random_start=False,
                )
            if eval_max_steps is not None and eval_max_steps > 0:
                eval_env = TimeLimit(eval_env, max_episode_steps=int(eval_max_steps))
            eval_env = Monitor(eval_env)

        # 安全性檢查：評估資料長度需大於 window_size+1，否則跳過評估
        try:
            eval_steps_available = len(eval_env.unwrapped.df) - eval_env.unwrapped.window_size - 1
        except Exception:
            eval_steps_available = 0

        if eval_steps_available <= 0:
            print("[評估跳過] 評估資料不足（長度 <= window_size+1），已略過本次評估以避免錯誤")
        else:
            # 使用自訂可重啟評估回調
            eval_callback = RestartingEvalCallback(
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=min(eval_episodes, 3) if eval_steps_available < 100 else eval_episodes,
                deterministic=True,
                verbose=1
            )
            callbacks.append(eval_callback)
    
    # 開始訓練
    print(f"\n{'='*60}")
    print(f"開始訓練 - 總步數: {total_timesteps}")
    print(f"{'='*60}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=None,
            progress_bar=True
        )
    except Exception as e:
        print("\n[錯誤位置] 訓練 (learn) 發生例外，即將拋出")
        raise
    
    # 保存最終模型
    final_model_path = model_dir / 'final_model.zip'
    model.save(str(final_model_path))
    print(f"\n最終模型已保存至: {final_model_path}")
    
    return model


def main() -> None:
    """主函數"""
    parser = argparse.ArgumentParser(description='SAC 交易模型訓練（SB3）')
    
    # 訓練參數（支持回合或步數）
    parser.add_argument('--timesteps', type=int, default=2000000,
                       help='總訓練步數（若未提供，將使用回合模式）')
    parser.add_argument('--random_start', action='store_true',
                       help='啟用回合隨機起點（每回合從隨機時間開始）')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'quick_test'],
                       help='運行模式')
    parser.add_argument('--n_envs', type=int, default=16,
                       help='並行環境數量（>1 啟用多進程）')
    
    # 數據和路徑
    parser.add_argument('--data', type=str, 
                       default='./Data/BTCUSDT_futures_volume_5years_5min.csv',
                       help='訓練數據路徑')
    parser.add_argument('--start_date', type=str, default='2025-01-01',
                       help='訓練資料開始日期（YYYY-MM-DD 或可解析字串）')
    parser.add_argument('--end_date', type=str, default='2025-06-30',
                       help='訓練資料結束日期（YYYY-MM-DD 或可解析字串）')
    parser.add_argument('--model_dir', type=str, default='./models',
                       help='模型保存目錄')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日誌目錄')
    
    # 模型參數
    parser.add_argument('--load_model', type=str, default=None,
                       help='加載已有模型路徑（繼續訓練）')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='訓練設備')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='學習率')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='批次大小')
    parser.add_argument('--buffer_size', type=int, default=10_0000,
                       help='經驗回放緩衝區大小')
    
    # 環境參數
    parser.add_argument('--margin_mode', type=str, default='isolated',
                       choices=['isolated', 'cross'],# isolated: 隔離保證金(逐倉), cross: 交叉保證金(全倉)
                       help='保證金模式')
    parser.add_argument('--leverage', type=float, default=10.0,
                       help='槓桿倍數')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                       help='初始資金')
    parser.add_argument('--min_balance', type=float, default=100.0,
                       help='最小資金')
    parser.add_argument('--window_size', type=int, default=288 * 7,#5min bars * 24 hours * 7 days = 288 bars * 7 days = 2016 bars
                       help='觀察窗口大小')
    parser.add_argument('--reward_mode', type=str, default='delta_equity',
                       choices=['delta_equity', 'pct', 'log'],
                       help='獎勵模式')
    
    # 評估資料設定
    parser.add_argument('--eval_use_last_month', action='store_true',
                       help='使用最新1個月作為評估資料集（自動計算日期範圍）')
    parser.add_argument('--eval_start_date', type=str, default='2025-07-01',
                       help='評估資料開始日期（YYYY-MM-DD）')
    parser.add_argument('--eval_end_date', type=str, default='2025-07-31',
                       help='評估資料結束日期（YYYY-MM-DD）')
    parser.add_argument('--eval_max_steps', type=int, default=1000,
                       help='每個評估回合的最大片長（步數），超過即截斷')
    parser.add_argument('--eval_freq', type=int, default=100000,
                       help='評估頻率（步數）')
    args = parser.parse_args()
    
    # 快速測試模式
    if args.mode == 'quick_test':
        print("\n=== 快速測試模式 ===")
        args.timesteps = 10000
        args.buffer_size = 10000
        args.window_size = 100
        print("已調整參數為快速測試模式\n")
    
    print("\n" + "="*60)
    print("SAC 交易模型訓練系統（Stable-Baselines3）")
    print("="*60 + "\n")
    
    try:
        # 創建環境
        # 1) timesteps 模式：一次性以總步數訓練

        if args.timesteps is not None:
            env = create_environment(
                data_path=args.data,
                initial_balance=args.initial_balance,
                min_balance=args.min_balance,
                leverage=args.leverage,
                window_size=args.window_size,
                reward_mode=args.reward_mode,
                start_date=args.start_date,
                end_date=args.end_date,
                random_start=args.random_start,
            )

            if args.eval_use_last_month:
                # 從來源 CSV 推算最新日期
                df_tmp = pd.read_csv(args.data)
                dt_col = None
                for candidate in ('datetime', 'date', 'time', 'timestamp'):
                    if candidate in df_tmp.columns:
                        dt_col = candidate
                        break
                if dt_col is not None:
                    df_tmp[dt_col] = pd.to_datetime(df_tmp[dt_col])
                    max_dt = df_tmp[dt_col].max()
                    eval_end_date = max_dt.strftime('%Y-%m-%d')
                    eval_start_date = (max_dt - timedelta(days=30)).strftime('%Y-%m-%d')
                    print(f"使用最新1個月作評估: {eval_start_date} ~ {eval_end_date}")
                else:
                    print("警告: 資料無日期欄位，無法自動計算最新1個月評估區間")
            else:
                eval_start_date = args.eval_start_date
                eval_end_date = args.eval_end_date

            model = train_sac(
                env=env,
                total_timesteps=args.timesteps,#總訓練步數
                learning_rate=args.lr,#學習率
                buffer_size=args.buffer_size,#經驗回放緩衝區大小
                batch_size=args.batch_size,#批次大小
                model_dir=args.model_dir,#模型保存目錄
                log_dir=args.log_dir,#日誌目錄
                device=args.device,#訓練設備
                load_model=args.load_model,#加載已有模型路徑
                n_envs=args.n_envs,
                train_data_path=args.data,
                eval_data_path=args.data,#評估數據路徑
                eval_start_date=eval_start_date,#評估資料開始日期
                eval_end_date=eval_end_date,#評估資料結束日期
                eval_max_steps=args.eval_max_steps,#每個評估回合的最大片長（步數），超過即截斷
                eval_freq=args.eval_freq#評估頻率（步數）
            )
       
        print(f"\n{'='*60}")
        print("訓練完成！")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n\n訓練被用戶中斷")
        
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

