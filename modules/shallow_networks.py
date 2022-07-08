import torch
import numpy as np


class LayerNorm(torch.nn.Module):
    """
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP_vectornet(torch.nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP_vectornet, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = torch.nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.relu(hidden_states)
        return hidden_states


class DecoderResCat(torch.nn.Module):
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP_vectornet(in_features, hidden_size)
        self.fc = torch.nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat(
            [hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class TimeEncode(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        # t has shape [batch_size, seq_len]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        t = t.unsqueeze(dim=2)

        # output has shape [batch_size, seq_len, dimension]
        output = torch.cos(self.w(t))

        return output


class TimeFuser(torch.nn.Module):
    def __init__(self, hidden_size):
        super(TimeFuser, self).__init__()

        self.agent_layer = DecoderResCat(hidden_size, hidden_size, hidden_size)
        self.time_encoder = TimeEncode(hidden_size)

    def forward(self, hidden_states, timestamps, lane_polyline_indexes):
        # hidden_states.shape = (batch_size, max_poly_num, hidden_size)
        # timestamps.shape = (batch_size)
        timestamps = torch.unsqueeze(timestamps, dim=1)
        # timestamps.shape = (batch_size, 1)
        timestamps = self.time_encoder(timestamps)
        # timestamps.shape = (batch_size, 1, hidden_size)
        polyline_num = hidden_states.shape[1]
        timestamps = torch.repeat_interleave(
            timestamps, repeats=polyline_num, dim=1)
        # timestamps.shape = (batch_size, max_poly_num, hidden_size)

        hidden_states_time = timestamps + hidden_states

        # agent mask generation
        agent_mask = torch.zeros(hidden_states.shape,
                                 device=hidden_states.device)
        # lane_mask = torch.zeros(hidden_states.shape,
        #                         device=hidden_states.device)
        for idx, lane_idx in enumerate(lane_polyline_indexes):
            agent_mask[idx, :lane_idx] = 1
            # lane_mask[idx, lane_idx:] = 1

        hidden_states_time *= agent_mask
        hidden_states_time = self.agent_layer(hidden_states_time)

        hidden_states += hidden_states_time
        # hidden_states_time = self.agent_layer(hidden_states_time)
        # hidden_states = hidden_states*lane_mask + hidden_states_time*agent_mask

        return hidden_states
