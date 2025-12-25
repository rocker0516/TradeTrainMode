import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import logging
from collections import deque
import shutil

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from Env.trading_env import TradingEnvironment
from Env.wrappers import ActionRepeatWrapper, ActionSmoothClipWrapper
from Train.sac_lagrangian import SACLagrangianAgent
from Train.buffer import ReplayBuffer
from Train.cost import CombinedCostCalculator
from Train.config import Config
from Train.curriculum import WinRateFeeCurriculum

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

# Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)


def cleanup_step_logs():
    """Clear step log directory at training start to avoid stale logs."""
    if getattr(Config, "STEP_LOG_ENABLED", False):
        log_dir = getattr(Config, "STEP_LOG_DIR", "step_logs")
        if os.path.exists(log_dir):
            try:
                shutil.rmtree(log_dir)
                logger.info(f"Cleared step log directory: {log_dir}")
            except Exception as e:
                logger.warning(f"Failed to clear step log directory {log_dir}: {e}")
        os.makedirs(log_dir, exist_ok=True)

def make_env(rank, df, seed=0):
    def _init():
        env = TradingEnvironment(
            df=df,
            env_id=rank
        )
        # Apply Action Repeat Wrapper
        if hasattr(Config, 'ACTION_REPEAT') and Config.ACTION_REPEAT > 1:
            env = ActionRepeatWrapper(env, repeat=Config.ACTION_REPEAT)

        # 先裁剪持倉上限，再做動作平滑，降低高頻翻倉/換手
        env = ActionSmoothClipWrapper(
            env,
            max_position_pct=getattr(Config, "MAX_POSITION_PCT", 0.5),
            smoothing_alpha=getattr(Config, "ACTION_SMOOTHING_ALPHA", 0.3)
        )
            
        env.reset(seed=seed + rank)
        return env
    return _init

def format_dashboard(global_step, total_episodes, fps, stats, metrics, costs, cost_limits, *, current_fee_rate: float):
    """Generates a clean periodic dashboard string"""
    width = 60
    
    # Calculate Statistics
    # Avg Profit 僅計算「成功存活至資料結束」的回合
    survived_profits = stats.get('survived_profits', [])
    survived_balances = stats.get('survived_balances', [])
    if survived_profits:
        avg_profit = np.mean(survived_profits)
        std_profit = np.std(survived_profits)
    else:
        # 若目前沒有任何存活回合，退而使用全部回合做參考（避免顯示全 0 誤導）
        avg_profit = np.mean(stats['profits']) if stats['profits'] else 0.0
        std_profit = np.std(stats['profits']) if stats['profits'] else 0.0
    # Balance 也必須對齊同一套樣本集合，否則 Avg Profit(存活) 與 Avg Balance(全部) 會數學上對不起來。
    avg_bal_all = np.mean(stats['balances']) if stats['balances'] else 0.0
    if survived_profits and survived_balances:
        avg_bal = np.mean(survived_balances)
    else:
        avg_bal = avg_bal_all
    # 透過 ActionRepeatWrapper，環境實際步數 = 記錄的決策步數 * ACTION_REPEAT
    avg_len = (np.mean(stats['lengths']) * Config.ACTION_REPEAT) if stats['lengths'] else 0.0
    min_len = (np.min(stats['lengths']) * Config.ACTION_REPEAT) if stats['lengths'] else 0.0
    max_len = (np.max(stats['lengths']) * Config.ACTION_REPEAT) if stats['lengths'] else 0.0
    avg_fees = np.mean(stats['fees']) if stats['fees'] else 0.0
    avg_trades = np.mean(stats['trades']) if stats['trades'] else 0.0
    avg_longs = np.mean(stats['longs']) if stats['longs'] else 0.0
    avg_shorts = np.mean(stats['shorts']) if stats['shorts'] else 0.0
    avg_sl = np.mean(stats['sl_counts']) if stats['sl_counts'] else 0.0
    # Stop-loss metrics:
    # - max_stop_loss_dist_pct: max distance between entry and stop price (unleveraged), printed as percent.
    # - max_sl_hit_rate: max stop-loss hit rate per episode (= sl_count / total_trades), printed as percent.
    max_stop_loss_dist_pct = np.max(stats.get('max_stop_loss_dists', [])) if stats.get('max_stop_loss_dists') else 0.0
    max_sl_hit_rate = np.max(stats.get('sl_rates', [])) if stats.get('sl_rates') else 0.0
    avg_missing_sl_rate = np.mean(stats.get('missing_sl_rates', [])) if stats.get('missing_sl_rates') else 0.0
    avg_abs_sl_gap = np.mean(stats.get('abs_sl_gap_means', [])) if stats.get('abs_sl_gap_means') else 0.0
    avg_pos_pct = np.mean(stats.get('pos_pct_means', [])) if stats.get('pos_pct_means') else 0.0
    avg_abs_pos_pct = np.mean(stats.get('abs_pos_pct_means', [])) if stats.get('abs_pos_pct_means') else 0.0
    avg_pos_time = np.mean(stats.get('pos_time_rates', [])) if stats.get('pos_time_rates') else 0.0
    avg_conv_bonus = np.mean(stats.get('conviction_bonus_means', [])) if stats.get('conviction_bonus_means') else 0.0
    avg_conv_active = np.mean(stats.get('conviction_active_rates', [])) if stats.get('conviction_active_rates') else 0.0
    
    # New Metric: Avg Max Trade Loss
    avg_max_trade_loss = np.mean(stats['max_trade_losses']) if stats['max_trade_losses'] else 0.0

    wins = [p for p in stats['profits'] if p > 0]
    win_rate = (len(wins) / len(stats['profits']) * 100) if stats['profits'] else 0.0
    
    total_eps_in_window = len(stats['reasons'])
    reason_counts = {}
    
    # Explicitly track these reasons to ensure they show up even if 0%
    known_reasons = ['liq_triggered', 'balance_insufficient', 'data_exhausted']
    
    for r in stats['reasons']:
        if r not in known_reasons:
            known_reasons.append(r)
        reason_counts[r] = reason_counts.get(r, 0) + 1
    
    # Format String
    lines = []
    lines.append("-" * width)
    lines.append(f"| Step: {global_step:,} | Ep: {total_episodes:,} | Prog: {global_step/Config.TOTAL_TIMESTEPS:.1%} | FPS: {int(fps)}".ljust(width-1) + "|")
    lines.append("-" * width)
    
    lines.append(f"| Account Performance (Last {len(stats['profits'])} Episodes):".ljust(width-1) + "|")# Last {len(stats['profits'])} Episodes 是最後幾集的平均收益
    lines.append(f"|   Avg Profit:      {avg_profit:+.2f}%  (± {std_profit:.1f}%)".ljust(width-1) + "|")
    if survived_profits and survived_balances:
        lines.append(f"|   Avg Balance:     {avg_bal:,.2f}  (All: {avg_bal_all:,.2f})".ljust(width-1) + "|")
    else:
        lines.append(f"|   Avg Balance:     {avg_bal_all:,.2f}".ljust(width-1) + "|")
    # fee_rate 單位沿用專案既有：0.005 代表 0.005%（taker fee）
    lines.append(f"|   Avg Fees:        {avg_fees:.2f} | FeeRate: {float(current_fee_rate):.6f}%".ljust(width-1) + "|")
    lines.append(f"|   Avg Ep Length:   {avg_len:.0f} steps (min {min_len:.0f}, max {max_len:.0f})".ljust(width-1) + "|")
    lines.append(f"|   Avg Trades:      {avg_trades:.1f} (L:{avg_longs:.1f}/S:{avg_shorts:.1f})".ljust(width-1) + "|")
    lines.append(f"|   Avg Position:    {avg_pos_pct*100:+.1f}% | Avg |Pos|: {avg_abs_pos_pct*100:.1f}% | PosTime: {avg_pos_time*100:.1f}%".ljust(width-1) + "|")
    lines.append(f"|   ConvBonus:      {avg_conv_bonus:.4f} | ActiveRate: {avg_conv_active*100:.1f}%".ljust(width-1) + "|")
    lines.append(f"|   Avg StopLoss:    {avg_sl:.1f}".ljust(width-1) + "|")
    lines.append(f"|   Max StopLoss%:   {max_stop_loss_dist_pct*100:.1f}%".ljust(width-1) + "|")
    lines.append(f"|   Max SL HitRate:  {max_sl_hit_rate*100:.1f}%".ljust(width-1) + "|")
    lines.append(f"|   Max Trade Loss:  {avg_max_trade_loss:+.2f}%".ljust(width-1) + "|")
    lines.append(f"|   Win Rate:        {win_rate:.1f}%".ljust(width-1) + "|")
    lines.append("|".ljust(width-1) + "|")
    
    lines.append("| Termination Reasons:".ljust(width-1) + "|")
    for reason in sorted(known_reasons):
        count = reason_counts.get(reason, 0)
        pct = (count / total_eps_in_window * 100) if total_eps_in_window > 0 else 0
        lines.append(f"|   {reason.ljust(15)}: {pct:.1f}%".ljust(width-1) + "|")
    lines.append("|".ljust(width-1) + "|")

    lines.append("| Training Metrics:".ljust(width-1) + "|")
    lines.append(f"|   Actor Loss:      {metrics.get('loss/actor', 0):.3f}".ljust(width-1) + "|")
    lines.append(f"|   Critic Loss:     {metrics.get('loss/critic', 0):.3f}".ljust(width-1) + "|")
    lines.append(f"|   Alpha (Ent):     {metrics.get('val/alpha', 0):.3f}".ljust(width-1) + "|")
    lines.append(f"|   Avg Q (Reward):  {metrics.get('val/avg_q', 0):.1f}".ljust(width-1) + "|")
    lines.append("|".ljust(width-1) + "|")
    
    lines.append("| Safety Constraints:".ljust(width-1) + "|")
    num_constraints = len(cost_limits)
    for i in range(num_constraints):
        avg_q_c = metrics.get(f'val/avg_cost_q_{i+1}', 0.0)
        lam = metrics.get(f'val/lambda_{i+1}', 0.0)
        limit = float(cost_limits[i])
        lines.append(f"|   C{i+1}: Est={avg_q_c:.3f} | Lim={limit:.2f} | λ={lam:.2f}".ljust(width-1) + "|")
    
    # --- Extra diagnostics (console) ---
    gate = metrics.get("val/lambda_gate", 0.0)
    c4_ema = metrics.get("val/cost_ema_4", 0.0)
    c4_viol_ema = metrics.get("val/cost_violation_ema_4", 0.0)
    lines.append(f"|   Gate={gate:.3f} | C4_EMA={c4_ema:.3f} | C4_ViolEMA={c4_viol_ema:+.3f}".ljust(width-1) + "|")
    # mean|sl_gap| uses only steps with position+has_stop; show 6 decimals to avoid rounding-to-zero confusion
    lines.append(f"|   SL_missing_rate={avg_missing_sl_rate*100:.1f}% | mean|sl_gap|={avg_abs_sl_gap:.6f}".ljust(width-1) + "|")
        
    lines.append("-" * width)
    return "\n".join(lines)

def train():
    logger.info("Initializing Training...")
    cleanup_step_logs()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"runs/sac_lagrangian_cnn/{timestamp}"
    writer = SummaryWriter(run_name)
    model_dir = f"models/{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    logger.info(f"TensorBoard: {run_name}")
    logger.info(f"Models: {model_dir}")
    logger.info(f"Config: {Config.__dict__}")

    # 1. Load Data
    logger.info(f"Loading data from {Config.DATA_PATH}...")
    df = pd.read_csv(Config.DATA_PATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # 2. Initialize Parallel Environments
    num_envs = Config.NUM_ENVS
    env_fns = [make_env(i, df, Config.SEED) for i in range(num_envs)]
    
    if num_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    
    # Apply VecNormalize to stabilize inputs and reward (Running Mean/Var)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=Config.GAMMA)

    # --- Fee Curriculum (stage-based by win rate) ---
    fee_curriculum = None
    if bool(getattr(Config, "FEE_CURRICULUM_ENABLED", False)):
        fee_curriculum = WinRateFeeCurriculum(
            base_fee_rate=float(getattr(Config, "TRANSACTION_FEE", 0.0)),
            initial_mult=float(getattr(Config, "FEE_CURRICULUM_INITIAL_MULT", 0.0)),
            step_mult=float(getattr(Config, "FEE_CURRICULUM_STEP_MULT", 0.1)),
            threshold=float(getattr(Config, "FEE_CURRICULUM_WINRATE_THRESHOLD", 0.5)),
            avg_profit_threshold_pct=(
                None
                if getattr(Config, "FEE_CURRICULUM_AVG_PROFIT_THRESHOLD_PCT", None) is None
                else float(getattr(Config, "FEE_CURRICULUM_AVG_PROFIT_THRESHOLD_PCT"))
            ),
            min_episodes=int(getattr(Config, "FEE_CURRICULUM_MIN_EPISODES", 50)),
            min_episodes_between_advances=int(getattr(Config, "FEE_CURRICULUM_MIN_EPISODES_BETWEEN_ADVANCES", 50)),
        )
        try:
            # VecNormalize 會 forward env_method 到 underlying VecEnv（含 Subproc/Dummy）
            env.env_method("set_fee_rate", float(fee_curriculum.fee_rate))
            logger.info(
                f"Fee curriculum initialized: fee_mult={fee_curriculum.fee_mult:.2f}, fee_rate={fee_curriculum.fee_rate:.6f}"
            )
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Fee curriculum enabled but failed to set initial fee_rate: {e}")
    
    # Get dimensions
    temp_env = make_env(0, df)()
    obs_sample, _ = temp_env.reset()
    price_seq_shape = obs_sample['price_seq'].shape
    daily_seq_shape = obs_sample['daily_seq'].shape # Get daily seq shape
    
    # Calculate total state dim from parts
    state_dim = (
        obs_sample['account_state'].shape[0] +
        obs_sample['time_state'].shape[0] +
        obs_sample['rhythm_state'].shape[0] +
        obs_sample['cost_state'].shape[0] +
        obs_sample['market_state'].shape[0]
    )
    
    action_dim = temp_env.action_space.shape[0]
    del temp_env
    
    logger.info(f"Obs Space: PriceSeq {price_seq_shape}, DailySeq {daily_seq_shape}, StateVec {state_dim}")
    logger.info(f"Action Space: {action_dim}")
    
    # 3. Initialize Agent
    agent = SACLagrangianAgent(
        price_input_channels=price_seq_shape[1],
        price_window_size=price_seq_shape[0],
        daily_input_channels=daily_seq_shape[1], # Pass daily channels
        daily_window_size=daily_seq_shape[0],    # Pass daily window size
        state_dim=state_dim,
        action_dim=action_dim,
        cost_limits=Config.COST_LIMITS,
        device=torch.device(Config.DEVICE),
        gamma=Config.GAMMA,
        tau=Config.TAU,
        lr=Config.LR,
        use_lagrangian=True,
        lagrangian_lr=getattr(Config, "LAGRANGIAN_LR", 0.05),
    )
    # 保存 base limits（供退火使用）
    base_cost_limits = np.array(Config.COST_LIMITS, dtype=np.float32)
    
    # 4. Initialize Buffer
    replay_buffer = ReplayBuffer(
        capacity=Config.BUFFER_SIZE,
        price_seq_shape=price_seq_shape,
        state_dim=state_dim,
        action_dim=action_dim,
        cost_dim=len(Config.COST_LIMITS),
        device=torch.device(Config.DEVICE),
        daily_seq_shape=daily_seq_shape # Pass daily seq shape
    )
    
    cost_calculator = CombinedCostCalculator(num_envs=num_envs)
    
    # Reset Envs
    obs = env.reset() 
    initial_balances = [Config.INITIAL_BALANCE] * num_envs
    cost_calculator.reset(range(num_envs), initial_balances)
    
    # Metrics Tracking
    episode_rewards = np.zeros(num_envs)
    episode_costs = np.zeros((num_envs, len(Config.COST_LIMITS)))
    episode_lengths = np.zeros(num_envs)
    # Risk diagnostics tracking (per-episode, per-env)
    episode_pos_steps = np.zeros(num_envs, dtype=np.int64)
    episode_missing_sl_steps = np.zeros(num_envs, dtype=np.int64)
    episode_has_stop_steps = np.zeros(num_envs, dtype=np.int64)
    episode_abs_sl_gap_sum = np.zeros(num_envs, dtype=np.float64)
    # Inventory diagnostics tracking (per-episode, per-env)
    episode_pos_pct_sum = np.zeros(num_envs, dtype=np.float64)
    episode_abs_pos_pct_sum = np.zeros(num_envs, dtype=np.float64)
    # Conviction bonus diagnostics (per-episode, per-env)
    episode_conv_bonus_sum = np.zeros(num_envs, dtype=np.float64)
    episode_conv_active_steps = np.zeros(num_envs, dtype=np.int64)
    
    # Dashboard Stats (Rolling Window)
    stats_window = 100
    stats = {
        'profits': deque(maxlen=stats_window),  # 所有回合的收益（含爆倉/死亡）
        'survived_profits': deque(maxlen=stats_window),  # 僅「存活至資料結束」回合的收益
        'survived_balances': deque(maxlen=stats_window),  # 僅「存活至資料結束」回合的期末資金
        'balances': deque(maxlen=stats_window),
        'lengths': deque(maxlen=stats_window),
        'reasons': deque(maxlen=stats_window),
        'fees': deque(maxlen=stats_window),
        'trades': deque(maxlen=stats_window),
        'longs': deque(maxlen=stats_window),
        'shorts': deque(maxlen=stats_window),
        'sl_counts': deque(maxlen=stats_window),
        # stop-loss diagnostics
        # max_stop_loss_dists: per-episode max(|entry - stop| / entry) in [0, +inf), printed as percent.
        'max_stop_loss_dists': deque(maxlen=stats_window),
        # sl_rates: per-episode ratio = episode_stop_loss_count / total_trades (0..1). Printed as percent in dashboard.
        'sl_rates': deque(maxlen=stats_window),
        'max_trade_losses': deque(maxlen=stats_window),
        # risk diagnostics (only meaningful when has position)
        'missing_sl_rates': deque(maxlen=stats_window),
        'abs_sl_gap_means': deque(maxlen=stats_window),
        # inventory diagnostics
        'pos_pct_means': deque(maxlen=stats_window),
        'abs_pos_pct_means': deque(maxlen=stats_window),
        'pos_time_rates': deque(maxlen=stats_window),
        # conviction bonus diagnostics
        'conviction_bonus_means': deque(maxlen=stats_window),
        'conviction_active_rates': deque(maxlen=stats_window),
    }
    
    logger.info("Starting training loop...")
    
    global_step = 0
    total_episodes = 0
    last_summary_step = 0
    summary_interval = Config.LOG_INTERVAL # Print dashboard every N steps
    start_time = time.time()
    metrics = {} # Store latest metrics
    
    # Initialize tqdm manually to avoid conflict
    pbar = tqdm(total=Config.TOTAL_TIMESTEPS, desc="Training", unit="step", dynamic_ncols=True)
    
    while global_step < Config.TOTAL_TIMESTEPS:
        # Select Action
        if global_step < Config.LEARNING_STARTS:
            actions = np.array([env.action_space.sample() for _ in range(num_envs)])
        else:
            # Batch inference
            actions = agent.select_action_batch(obs)

        # Step
        next_obs, rewards, dones, infos = env.step(actions)
        costs = cost_calculator.calculate_costs(infos) 
        
        # Vectorized Stats Updates
        episode_rewards += rewards
        episode_costs += costs
        episode_lengths += 1
        
        # --- Risk diagnostics (from env infos) ---
        # These are computed from raw env risk_signals (not from mixed C4 cost),
        # so we can diagnose whether C4 is driven by missing SL or proximity.
        # Env info may not include a reliable has_position flag; infer from maintenance_margin (>0 => has position)
        maint_margins = np.array([float(info.get("maintenance_margin", 0.0)) for info in infos], dtype=np.float32)
        has_pos = maint_margins > 1e-8
        risks = [info.get("risk_signals") or {} for info in infos]
        missing_sl = np.array([float(r.get("stop_loss_missing", 0.0)) for r in risks], dtype=np.float32) > 0.5
        sl_gap = np.array([float(r.get("sl_gap_pct", 0.0)) for r in risks], dtype=np.float32)
        episode_pos_steps += has_pos.astype(np.int64)
        episode_missing_sl_steps += (has_pos & missing_sl).astype(np.int64)
        has_stop = has_pos & (~missing_sl)
        episode_has_stop_steps += has_stop.astype(np.int64)
        episode_abs_sl_gap_sum += (np.abs(sl_gap) * has_stop.astype(np.float32))

        # --- Inventory diagnostics (from env infos) ---
        pos_pct_arr = np.array([float(info.get("position_pct", 0.0)) for info in infos], dtype=np.float32)
        abs_pos_pct_arr = np.array([float(info.get("abs_position_pct", abs(p))) for info, p in zip(infos, pos_pct_arr)], dtype=np.float32)
        episode_pos_pct_sum += pos_pct_arr.astype(np.float64)
        episode_abs_pos_pct_sum += abs_pos_pct_arr.astype(np.float64)

        # --- Conviction bonus diagnostics (from env infos) ---
        conv_bonus_arr = np.array([float(info.get("conviction_bonus", 0.0)) for info in infos], dtype=np.float32)
        conv_active_arr = np.array([bool(info.get("conviction_active", False)) for info in infos], dtype=np.bool_)
        episode_conv_bonus_sum += conv_bonus_arr.astype(np.float64)
        episode_conv_active_steps += conv_active_arr.astype(np.int64)
        
        # Prepare next observations (Handle terminal states)
        # Conditional copy: Only deep copy if there are done envs that need patching
        if np.any(dones):
            real_next_obs = {k: v.copy() for k, v in next_obs.items()}
            # Handle Dones (Logging & Term Obs Injection)
            done_indices = np.where(dones)[0]
            total_episodes += len(done_indices)
            for idx in done_indices:
                info = infos[idx]
                
                # Overwrite real_next_obs with terminal observation if available
                if 'terminal_observation' in info:
                    term_obs = info['terminal_observation']
                    for k in real_next_obs:
                        real_next_obs[k][idx] = term_obs[k]
                
                # Gather Stats
                final_bal = info.get('final_balance', Config.INITIAL_BALANCE)
                profit_pct = info.get('profit_rate', 0.0)
                term_reason = info.get('termination_reason', 'unknown')
                
                total_fees = info.get('total_fees', 0.0)
                long_entries = info.get('long_entry_count', 0)
                short_entries = info.get('short_entry_count', 0)
                sl_count = info.get('episode_stop_loss_count', 0)
                total_trades = long_entries + short_entries
                max_trade_loss_pct = info.get('max_single_trade_loss_pct', 0.0)
                sl_rate = (float(sl_count) / float(total_trades)) if int(total_trades) > 0 else 0.0
                max_stop_loss_dist_pct = float(info.get("max_stop_loss_distance_pct", 0.0))

                stats['profits'].append(profit_pct)
                # 僅將「成功走完資料」的回合收益，記錄到 survived_profits
                if term_reason == 'data_exhausted':
                    stats['survived_profits'].append(profit_pct)
                    stats['survived_balances'].append(final_bal)
                stats['balances'].append(final_bal)
                stats['lengths'].append(episode_lengths[idx])
                stats['reasons'].append(term_reason)
                stats['fees'].append(total_fees)
                stats['trades'].append(total_trades)
                stats['longs'].append(long_entries)
                stats['shorts'].append(short_entries)
                stats['sl_counts'].append(sl_count)
                stats['max_stop_loss_dists'].append(max_stop_loss_dist_pct)
                stats['sl_rates'].append(float(sl_rate))
                stats['max_trade_losses'].append(max_trade_loss_pct)
                
                # Risk diagnostics (per episode)
                pos_steps = int(episode_pos_steps[idx])
                missing_steps = int(episode_missing_sl_steps[idx])
                has_stop_steps = int(episode_has_stop_steps[idx])
                missing_rate = (missing_steps / max(1, pos_steps)) if pos_steps > 0 else 0.0
                abs_sl_gap_mean = (float(episode_abs_sl_gap_sum[idx]) / max(1, has_stop_steps)) if has_stop_steps > 0 else 0.0
                stats['missing_sl_rates'].append(float(missing_rate))
                stats['abs_sl_gap_means'].append(float(abs_sl_gap_mean))

                # Inventory diagnostics (per episode)
                ep_len = float(max(1.0, episode_lengths[idx]))
                pos_pct_mean = float(episode_pos_pct_sum[idx] / ep_len)
                abs_pos_pct_mean = float(episode_abs_pos_pct_sum[idx] / ep_len)
                pos_time_rate = float(pos_steps / max(1, int(episode_lengths[idx])))  # uses has_pos from maint_margin>0
                stats['pos_pct_means'].append(pos_pct_mean)
                stats['abs_pos_pct_means'].append(abs_pos_pct_mean)
                stats['pos_time_rates'].append(pos_time_rate)

                # Conviction bonus diagnostics (per episode)
                conv_bonus_mean = float(episode_conv_bonus_sum[idx] / ep_len)
                conv_active_rate = float(episode_conv_active_steps[idx] / max(1.0, ep_len))
                stats['conviction_bonus_means'].append(conv_bonus_mean)
                stats['conviction_active_rates'].append(conv_active_rate)
                
                # TensorBoard (Per Episode)
                writer.add_scalar("rollout/episode_reward", episode_rewards[idx], global_step)
                writer.add_scalar("rollout/episode_len", episode_lengths[idx], global_step)
                writer.add_scalar("rollout/profit_rate", profit_pct, global_step)
                writer.add_scalar("rollout/total_fees", total_fees, global_step)
                writer.add_scalar("rollout/total_trades", total_trades, global_step)
                writer.add_scalar("rollout/sl_count", sl_count, global_step)
                writer.add_scalar("rollout/max_stop_loss_distance_pct", max_stop_loss_dist_pct, global_step)
                writer.add_scalar("rollout/sl_rate", float(sl_rate), global_step)
                writer.add_scalar("rollout/max_single_trade_loss", max_trade_loss_pct, global_step)
                writer.add_scalar("rollout/missing_sl_rate", float(missing_rate), global_step)
                writer.add_scalar("rollout/abs_sl_gap_mean", float(abs_sl_gap_mean), global_step)
                writer.add_scalar("rollout/pos_pct_mean", float(pos_pct_mean), global_step)
                writer.add_scalar("rollout/abs_pos_pct_mean", float(abs_pos_pct_mean), global_step)
                writer.add_scalar("rollout/pos_time_rate", float(pos_time_rate), global_step)
                writer.add_scalar("rollout/conviction_bonus_mean", float(conv_bonus_mean), global_step)
                writer.add_scalar("rollout/conviction_active_rate", float(conv_active_rate), global_step)
                
                # Reset Trackers
                episode_rewards[idx] = 0
                episode_costs[idx] = np.zeros(len(Config.COST_LIMITS))
                episode_lengths[idx] = 0
                episode_pos_steps[idx] = 0
                episode_missing_sl_steps[idx] = 0
                episode_has_stop_steps[idx] = 0
                episode_abs_sl_gap_sum[idx] = 0.0
                episode_pos_pct_sum[idx] = 0.0
                episode_abs_pos_pct_sum[idx] = 0.0
                episode_conv_bonus_sum[idx] = 0.0
                episode_conv_active_steps[idx] = 0
                cost_calculator.reset([idx], [Config.INITIAL_BALANCE])
        else:
             real_next_obs = next_obs

        # --- Sample Filtering (Quality Control) ---
        # 邏輯：過濾掉「無效動作」且「無顯著後果」的樣本，減少 Buffer 冗餘。
        # 條件：|Action| < 0.01 (幾乎不動) AND |Reward| < 0.01 (無損益) AND Not Done
        # 保留率：10% (即丟棄 90% 的這類樣本)
        
        # 1. 計算過濾掩碼 (True = 保留, False = 丟棄)
        # 動作幅度極小
        small_action = np.abs(actions).squeeze() < Config.FILTER_SMALL_ACTION_THRESHOLD
        # 獎勵回饋極小
        small_reward = np.abs(rewards) < Config.FILTER_SMALL_REWARD_THRESHOLD
        # 且不是結束狀態 (結束狀態必須保留)
        not_done = ~dones
        
        # 候選丟棄樣本
        candidates = small_action & small_reward & not_done
        
        # 隨機保留 10% 的候選樣本 (丟棄 90%)
        keep_probability = 1.0 - Config.FILTER_DROP_PROBABILITY
        keep_mask = ~candidates | (np.random.random(size=len(actions)) < keep_probability)
        
        # 2. 根據掩碼篩選數據
        if np.any(keep_mask):
            # 對 Dict obs 進行篩選
            filtered_obs = {k: v[keep_mask] for k, v in obs.items()}
            filtered_next_obs = {k: v[keep_mask] for k, v in real_next_obs.items()}
            
            # Batch Add to Buffer
            replay_buffer.add_batch(
                filtered_obs,
                actions[keep_mask],
                rewards[keep_mask],
                costs[keep_mask],
                filtered_next_obs,
                dones[keep_mask]
            )
            
        obs = next_obs
        global_step += num_envs
        
        # Update Progress Bar
        pbar.update(num_envs)
        
        # Update Agent
        if global_step >= Config.LEARNING_STARTS:
            # --- Cost limit annealing (避免早期 λ 爆炸) ---
            anneal_steps = float(getattr(Config, "COST_LIMIT_ANNEAL_STEPS", 0.0))
            mult_start = float(getattr(Config, "COST_LIMIT_MULT_START", 1.0))
            mult_end = float(getattr(Config, "COST_LIMIT_MULT_END", 1.0))
            if anneal_steps > 0 and hasattr(agent, "set_cost_limits"):
                frac = min(1.0, max(0.0, global_step / anneal_steps))
                mult = mult_start + (mult_end - mult_start) * frac
                agent.set_cost_limits((base_cost_limits * mult).tolist())

            # Update once per step (or adjust ratio)
            metrics = agent.update(replay_buffer, Config.BATCH_SIZE, global_step=global_step)
            
            # Fix logging frequency bug: ensure we log roughly every 100 steps
            # Since global_step jumps by num_envs, strict modulo 100 might fail
            if global_step % 100 < num_envs:
                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step)
        
        # Dashboard Summary
        if global_step - last_summary_step >= summary_interval: #每隔N步更新一次dashboard
            elapsed = time.time() - start_time
            fps = global_step / elapsed
            current_limits = agent.get_cost_limits() if hasattr(agent, "get_cost_limits") else np.array(Config.COST_LIMITS, dtype=np.float32)

            # Current fee_rate for dashboard (prefer curriculum; else try env; else fallback to Config)
            current_fee_rate = float(getattr(Config, "TRANSACTION_FEE", 0.0))
            if fee_curriculum is not None:
                current_fee_rate = float(fee_curriculum.fee_rate)
            else:
                try:
                    fee_rates = env.env_method("get_fee_rate")
                    if isinstance(fee_rates, (list, tuple)) and len(fee_rates) > 0:
                        current_fee_rate = float(np.mean([float(x) for x in fee_rates]))
                except (AttributeError, TypeError, ValueError):
                    pass

            dashboard = format_dashboard(
                global_step,
                total_episodes,
                fps,
                stats,
                metrics,
                episode_costs[0],
                current_limits,
                current_fee_rate=current_fee_rate,
            )
            
            # Clear pbar, print dashboard, then refresh pbar
            pbar.clear()
            print("\n" + dashboard + "\n") # Add newlines for spacing
            pbar.refresh()
            
            last_summary_step = global_step

            # --- Fee Curriculum evaluation (advance stage by win rate) ---
            if fee_curriculum is not None:
                # Align with dashboard: prefer survived_profits if available, else use all profits.
                profits_all = list(stats.get("profits", []))
                profits_for_eval = list(stats.get("survived_profits", [])) or profits_all

                if profits_for_eval:
                    wins = [p for p in profits_for_eval if p > 0]
                    win_rate = len(wins) / len(profits_for_eval)
                    avg_profit_pct = float(np.mean(profits_for_eval))
                else:
                    win_rate = 0.0
                    avg_profit_pct = 0.0

                new_fee_rate = fee_curriculum.maybe_advance(
                    win_rate=win_rate,
                    avg_profit_pct=avg_profit_pct,
                    total_episodes=total_episodes,
                )
                if new_fee_rate is not None:
                    try:
                        env.env_method("set_fee_rate", float(new_fee_rate))
                        writer.add_scalar("curriculum/fee_mult", float(fee_curriculum.fee_mult), global_step)
                        writer.add_scalar("curriculum/fee_rate", float(new_fee_rate), global_step)
                        logger.info(
                            f"Fee curriculum advanced: win_rate={win_rate:.3f}, avg_profit_pct={avg_profit_pct:.2f}%, "
                            f"fee_mult={fee_curriculum.fee_mult:.2f}, fee_rate={float(new_fee_rate):.6f}"
                        )
                    except (AttributeError, TypeError, ValueError) as e:
                        logger.warning(f"Fee curriculum advanced but failed to set fee_rate: {e}")
        
        # Save Model
        if global_step % 1_000_000 < num_envs: 
            torch.save(agent.actor.state_dict(), f"{model_dir}/actor_{global_step}.pth")
            env.save(f"{model_dir}/vec_normalize_{global_step}.pkl")
            
            # Prevent tqdm glitch by clearing line or using pbar.write
            msg = f"Model saved at step {global_step}"
            if 'pbar' in locals() and pbar is not None:
                pbar.write(msg)
            else:
                logger.info(msg)
            
    pbar.close()
    env.close()
    writer.close()
    logger.info("Training completed.")

if __name__ == "__main__":
    train()
