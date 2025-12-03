
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

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from Env.trading_env import TradingEnvironment
from Train.sac_lagrangian import SACLagrangianAgent
from Train.buffer import ReplayBuffer
from Train.cost import CombinedCostCalculator
from Train.config import Config

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

def make_env(rank, df, seed=0):
    def _init():
        env = TradingEnvironment(
            df=df,
            window_size=Config.WINDOW_SIZE,
            leverage=Config.LEVERAGE,
            initial_balance=Config.INITIAL_BALANCE,
            min_balance=Config.MIN_BALANCE,
            transaction_fee=Config.TRANSACTION_FEE,
            min_episode_steps=Config.MIN_EPISODE_STEPS,
            min_position_change=Config.MIN_POSITION_CHANGE,
            max_step_pos_change_pct=Config.MAX_STEP_POS_CHANGE_PCT,
            turnover_penalty=Config.REWARD_TURNOVER_PENALTY,
            dd_penalty_coef=Config.REWARD_DD_PENALTY,
            hold_bonus=Config.REWARD_HOLD_BONUS,
            random_start=True
        )
        env.reset(seed=seed + rank)
        return env
    return _init

def format_dashboard(global_step, fps, stats, metrics, costs, num_constraints):
    """Generates a clean periodic dashboard string"""
    width = 60
    
    # Calculate Statistics
    avg_profit = np.mean(stats['profits']) if stats['profits'] else 0.0
    std_profit = np.std(stats['profits']) if stats['profits'] else 0.0
    avg_bal = np.mean(stats['balances']) if stats['balances'] else 0.0
    avg_len = np.mean(stats['lengths']) if stats['lengths'] else 0.0
    
    avg_fees = np.mean(stats['fees']) if stats['fees'] else 0.0
    avg_trades = np.mean(stats['trades']) if stats['trades'] else 0.0
    avg_longs = np.mean(stats['longs']) if stats['longs'] else 0.0
    avg_shorts = np.mean(stats['shorts']) if stats['shorts'] else 0.0
    avg_sl = np.mean(stats['sl_counts']) if stats['sl_counts'] else 0.0
    
    # New Metric: Avg Max Trade Loss
    avg_max_trade_loss = np.mean(stats['max_trade_losses']) if stats['max_trade_losses'] else 0.0

    wins = [p for p in stats['profits'] if p > 0]
    win_rate = (len(wins) / len(stats['profits']) * 100) if stats['profits'] else 0.0
    
    total_eps = len(stats['reasons'])
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
    lines.append(f"| Step: {global_step:,} | Progress: {global_step/Config.TOTAL_TIMESTEPS:.1%} | FPS: {int(fps)}".ljust(width-1) + "|")
    lines.append("-" * width)
    
    lines.append(f"| Account Performance (Last {len(stats['profits'])} Episodes):".ljust(width-1) + "|")# Last {len(stats['profits'])} Episodes 是最後幾集的平均收益
    lines.append(f"|   Avg Profit:      {avg_profit:+.2f}%  (± {std_profit:.1f}%)".ljust(width-1) + "|")
    lines.append(f"|   Avg Balance:     {avg_bal:,.2f}".ljust(width-1) + "|")
    lines.append(f"|   Avg Fees:        {avg_fees:.2f}".ljust(width-1) + "|")
    lines.append(f"|   Avg Ep Length:   {avg_len:.0f} steps".ljust(width-1) + "|")
    lines.append(f"|   Avg Trades:      {avg_trades:.1f} (L:{avg_longs:.1f}/S:{avg_shorts:.1f})".ljust(width-1) + "|")
    lines.append(f"|   Avg StopLoss:    {avg_sl:.1f}".ljust(width-1) + "|")
    lines.append(f"|   Max Trade Loss:  {avg_max_trade_loss:+.2f}%".ljust(width-1) + "|")
    lines.append(f"|   Win Rate:        {win_rate:.1f}%".ljust(width-1) + "|")
    lines.append("|".ljust(width-1) + "|")
    
    lines.append("| Termination Reasons:".ljust(width-1) + "|")
    for reason in sorted(known_reasons):
        count = reason_counts.get(reason, 0)
        pct = (count / total_eps * 100) if total_eps > 0 else 0
        lines.append(f"|   {reason.ljust(15)}: {pct:.1f}%".ljust(width-1) + "|")
    lines.append("|".ljust(width-1) + "|")

    lines.append("| Training Metrics:".ljust(width-1) + "|")
    lines.append(f"|   Actor Loss:      {metrics.get('loss/actor', 0):.3f}".ljust(width-1) + "|")
    lines.append(f"|   Critic Loss:     {metrics.get('loss/critic', 0):.3f}".ljust(width-1) + "|")
    lines.append(f"|   Alpha (Ent):     {metrics.get('val/alpha', 0):.3f}".ljust(width-1) + "|")
    lines.append(f"|   Avg Q (Reward):  {metrics.get('val/avg_q', 0):.1f}".ljust(width-1) + "|")
    lines.append("|".ljust(width-1) + "|")
    
    lines.append("| Safety Constraints:".ljust(width-1) + "|")
    for i in range(num_constraints):
        avg_q_c = metrics.get(f'val/avg_cost_q_{i+1}', 0.0)
        lam = metrics.get(f'val/lambda_{i+1}', 0.0)
        limit = Config.COST_LIMITS[i]
        lines.append(f"|   C{i+1}: Est={avg_q_c:.3f} | Lim={limit:.2f} | λ={lam:.2f}".ljust(width-1) + "|")
        
    lines.append("-" * width)
    return "\n".join(lines)

def train():
    logger.info("Initializing Training...")
    
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
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=Config.GAMMA)
    
    # Get dimensions
    temp_env = make_env(0, df)()
    obs_sample, _ = temp_env.reset()
    price_seq_shape = obs_sample['price_seq'].shape
    
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
    
    logger.info(f"Obs Space: PriceSeq {price_seq_shape}, StateVec {state_dim}")
    logger.info(f"Action Space: {action_dim}")
    
    # 3. Initialize Agent
    agent = SACLagrangianAgent(
        price_input_channels=price_seq_shape[1],
        price_window_size=price_seq_shape[0],
        state_dim=state_dim,
        action_dim=action_dim,
        cost_limits=Config.COST_LIMITS,
        device=torch.device(Config.DEVICE),
        gamma=Config.GAMMA,
        tau=Config.TAU,
        lr=Config.LR,
        use_lagrangian=True
    )
    
    # 4. Initialize Buffer
    replay_buffer = ReplayBuffer(
        capacity=Config.BUFFER_SIZE,
        price_seq_shape=price_seq_shape,
        state_dim=state_dim,
        action_dim=action_dim,
        cost_dim=len(Config.COST_LIMITS),
        device=torch.device(Config.DEVICE)
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
    
    # Dashboard Stats (Rolling Window)
    stats_window = 100
    stats = {
        'profits': deque(maxlen=stats_window),
        'balances': deque(maxlen=stats_window),
        'lengths': deque(maxlen=stats_window),
        'reasons': deque(maxlen=stats_window),
        'fees': deque(maxlen=stats_window),
        'trades': deque(maxlen=stats_window),
        'longs': deque(maxlen=stats_window),
        'shorts': deque(maxlen=stats_window),
        'sl_counts': deque(maxlen=stats_window),
        'max_trade_losses': deque(maxlen=stats_window)
    }
    
    logger.info("Starting training loop...")
    
    global_step = 0
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
            actions = []
            for i in range(num_envs):
                single_obs = {k: v[i] for k, v in obs.items()}
                actions.append(agent.select_action(single_obs))
            actions = np.array(actions)

        # Step
        next_obs, rewards, dones, infos = env.step(actions)
        costs = cost_calculator.calculate_costs(infos) 
        
        for i in range(num_envs):
            real_next_obs = {k: next_obs[k][i] for k in next_obs}
            if dones[i]:
                if 'terminal_observation' in infos[i]:
                    term_obs = infos[i]['terminal_observation']
                    real_next_obs = term_obs
                    
            replay_buffer.add(
                {k: obs[k][i] for k in obs},
                actions[i],
                rewards[i],
                costs[i],
                real_next_obs,
                dones[i]
            )
            
            episode_rewards[i] += rewards[i]
            episode_costs[i] += costs[i]
            episode_lengths[i] += 1
            
            if dones[i]:
                # Gather stats for Dashboard
                final_bal = infos[i].get('final_balance', Config.INITIAL_BALANCE)
                profit_pct = infos[i].get('profit_rate', 0.0)
                term_reason = infos[i].get('termination_reason', 'unknown')
                
                total_fees = infos[i].get('total_fees', 0.0)
                long_entries = infos[i].get('long_entry_count', 0)
                short_entries = infos[i].get('short_entry_count', 0)
                sl_count = infos[i].get('episode_stop_loss_count', 0)
                total_trades = long_entries + short_entries
                max_trade_loss_pct = infos[i].get('max_single_trade_loss_pct', 0.0)

                stats['profits'].append(profit_pct)
                stats['balances'].append(final_bal)
                stats['lengths'].append(episode_lengths[i])
                stats['reasons'].append(term_reason)
                stats['fees'].append(total_fees)
                stats['trades'].append(total_trades)
                stats['longs'].append(long_entries)
                stats['shorts'].append(short_entries)
                stats['sl_counts'].append(sl_count)
                stats['max_trade_losses'].append(max_trade_loss_pct)
                
                # TensorBoard (Per Episode)
                writer.add_scalar("rollout/episode_reward", episode_rewards[i], global_step)
                writer.add_scalar("rollout/episode_len", episode_lengths[i], global_step)
                writer.add_scalar("rollout/profit_rate", profit_pct, global_step)
                writer.add_scalar("rollout/total_fees", total_fees, global_step)
                writer.add_scalar("rollout/total_trades", total_trades, global_step)
                writer.add_scalar("rollout/sl_count", sl_count, global_step)
                writer.add_scalar("rollout/max_single_trade_loss", max_trade_loss_pct, global_step)
                
                # Reset
                episode_rewards[i] = 0
                episode_costs[i] = np.zeros(len(Config.COST_LIMITS))
                episode_lengths[i] = 0
                cost_calculator.dd_costs[i].reset(Config.INITIAL_BALANCE)

        obs = next_obs
        global_step += num_envs
        
        # Update Progress Bar
        pbar.update(num_envs)
        
        # Update Agent
        if global_step >= Config.LEARNING_STARTS:
            # Update once per step (or adjust ratio)
            metrics = agent.update(replay_buffer, Config.BATCH_SIZE)
            
            if global_step % 100 == 0:
                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step)
        
        # Dashboard Summary
        if global_step - last_summary_step >= summary_interval:
            elapsed = time.time() - start_time
            fps = global_step / elapsed
            dashboard = format_dashboard(global_step, fps, stats, metrics, episode_costs[0], len(Config.COST_LIMITS))
            
            # Clear pbar, print dashboard, then refresh pbar
            pbar.clear()
            print("\n" + dashboard + "\n") # Add newlines for spacing
            pbar.refresh()
            
            last_summary_step = global_step
        
        # Save Model
        if global_step % 10000 < num_envs: 
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
