
import torch
import torch.nn as nn
from custom_transformer import KANTransformerEncoder
from utils import *

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_outputs, hidden_space, dropout_rate=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.transform_layer = nn.Linear(input_dim, hidden_space)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_space, nhead=num_heads,
            dropout=dropout_rate, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers,
            enable_nested_tensor=False
        )
        self.output_layer = nn.Linear(hidden_space, num_outputs)

    def forward(self, x):
        x = self.transform_layer(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.output_layer(x)
        return x

class TimeSeriesTransformer_ekan(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_outputs, hidden_space, dropout_rate=0.1, embed_dim=57, grid_size=32, degree=3, base_activation=nn.SiLU, use_residual_scaling=False):
        super(TimeSeriesTransformer_ekan, self).__init__()
        # 使用传入的、正确的input_dim
        self.transform_layer = nn.Linear(input_dim, hidden_space)
        self.transformer_encoder = KANTransformerEncoder(
            num_layers=num_layers,
            d_model=hidden_space,
            nhead=num_heads,
            dropout=dropout_rate,
            grid_size=grid_size,
            degree=degree,
            base_activation=base_activation,
            use_residual_scaling=use_residual_scaling  # 新增参数
        )
        self.output_layer = nn.Linear(hidden_space, num_outputs)

    def forward(self, x):
        x = self.transform_layer(x)
        x = self.transformer_encoder(x)
        x = self.output_layer(x[:, -1, :])
        return x
