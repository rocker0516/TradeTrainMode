import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math

from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

from trading_env import TradingEnvironment

class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding for Transformer models.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor using a Transformer Encoder.
    It processes the time-series observation from the environment.
    """
    def __init__(self, observation_space, features_dim: int = 128, n_head: int = 4, n_layers: int = 2):
        super().__init__(observation_space, features_dim)
        
        # The number of features in the observation (e.g., open, high, low, close, indicators, etc.)
        n_input_features = observation_space.shape[0]
        
        self.d_model = features_dim
        
        # 1. Input projection layer
        # Project the input features to the Transformer's model dimension
        self.input_proj = nn.Linear(n_input_features, self.d_model)
        
        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=self.d_model)
        
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_head, batch_first=False, dim_feedforward=self.d_model*4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 provides observations as (Batch, Features, SequenceLength)
        # We need to permute it to what PyTorch Transformer expects.
        
        # (B, C, L) -> (B, L, C)
        observations = observations.permute(0, 2, 1)
        
        # Project features to d_model: (B, L, C) -> (B, L, D_MODEL)
        x = self.input_proj(observations)
        
        # Permute for Transformer input: (B, L, D_MODEL) -> (L, B, D_MODEL)
        x = x.permute(1, 0, 2)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Aggregate the sequence into a single feature vector
        # We take the mean of the encoder's output sequence
        # (L, B, D_MODEL) -> (B, D_MODEL)
        features = x.mean(dim=0)
        
        return features


if __name__ == "__main__":
    # --- 1. Load Real Market Data ---
    print("Loading market data from CSV...")
    try:
        df = pd.read_csv('Data/BTCUSDT_futures_volume_5years_5min.csv')
    except FileNotFoundError:
        print("Error: 'Data/BTCUSDT_futures_volume_5years_5min.csv' not found.")
        print("Please make sure the CSV file is in the 'Data' subdirectory.")
        exit()

    # --- 2. Preprocess Data ---
    print("Preprocessing data...")
    
    # Assuming the first column is the timestamp
    # If your timestamp column has a different name, change 'timestamp' to the correct name.
    # Common names: 'timestamp', 'open_time', 'time'
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'open_time' in df.columns:
         df['open_time'] = pd.to_datetime(df['open_time'])
         df.set_index('open_time', inplace=True)
    else:
        print("Warning: No 'timestamp' or 'open_time' column found. Time features will be incorrect.")
        print("Continuing without time-based index.")

    # Standardize column names to lowercase
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)

    # Ensure all required columns are present
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one of the required columns: {required_cols}")
        exit()

    # Drop rows with any missing values
    df.dropna(inplace=True)
    
    # Keep only the necessary columns for the environment
    df = df[required_cols]

    print(f"Data loaded and preprocessed. Shape: {df.shape}")

    # Instantiate the trading environment
    env = TradingEnvironment(df=df)

    # --- 3. Configure the SAC Model with Transformer Policy ---
    print("Configuring SAC model with Transformer policy...")
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,  # Dimension of the feature vector from Transformer
            n_head=4,          # Number of heads in the multi-head attention
            n_layers=2         # Number of layers in the Transformer encoder
        ),
    )

    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./sac_transformer_trading_tensorboard/",
        buffer_size=100_000  # Set buffer size to avoid memory issues with large data
    )

    # --- 4. Train the Agent ---
    print("Starting training...")
    # For a real training session, consider increasing total_timesteps significantly (e.g., 1_000_000)
    model.learn(total_timesteps=100_000, log_interval=100)

    # --- 5. Save the Model ---
    print("Training finished. Saving model...")
    model.save("sac_transformer_trading_model_real_data")

    print("\nModel saved as sac_transformer_trading_model_real_data.zip")
    print("To monitor training, run: tensorboard --logdir ./sac_transformer_trading_tensorboard/")

    # --- Optional: How to load and use the model ---
    # print("\nLoading saved model...")
    # loaded_model = SAC.load("sac_transformer_trading_model")
    #
    # obs, info = env.reset()
    # for _ in range(500):
    #     action, _states = loaded_model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         print("Episode finished.")
    #         obs, info = env.reset() 