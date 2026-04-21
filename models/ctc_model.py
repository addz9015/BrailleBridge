import torch
import torch.nn as nn
from models.vocab import VOCAB_SIZE


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0, dropout=0.0, bias=True):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.rank > 0:
            self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
            nn.init.zeros_(self.lora_b)
        else:
            self.register_parameter('lora_a', None)
            self.register_parameter('lora_b', None)

    def forward(self, x):
        output = self.base(x)
        if self.rank <= 0:
            return output

        update = self.dropout(x) @ self.lora_a.t()
        update = update @ self.lora_b.t()
        return output + update * self.scaling


class BrailleCTCModel(nn.Module):
    def __init__(self, use_lora=False, lora_rank=8, lora_alpha=16.0, lora_dropout=0.0):
        super().__init__()
        self.use_lora = use_lora

        self.cnn_stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(0.10),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(0.10),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.cnn_tail = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(0.10),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.dropout = nn.Dropout(0.35)
        self.out_norm = nn.LayerNorm(512)
        if use_lora:
            self.fc = LoRALinear(512, VOCAB_SIZE, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        else:
            self.fc = nn.Linear(512, VOCAB_SIZE)

        # Encourage early non-blank exploration to avoid CTC all-blank collapse.
        with torch.no_grad():
            if use_lora:
                self.fc.base.bias.fill_(0.0)
                self.fc.base.bias[0] = -2.0
            else:
                self.fc.bias.fill_(0.0)
                self.fc.bias[0] = -2.0

    def lora_parameters(self):
        if not self.use_lora:
            return []
        return [param for name, param in self.named_parameters() if 'lora_' in name]

    def freeze_base_model(self):
        for param in self.parameters():
            param.requires_grad = False

        if self.use_lora:
            for param in self.lora_parameters():
                param.requires_grad = True
            for param in self.out_norm.parameters():
                param.requires_grad = True
            if self.fc.base.bias is not None:
                self.fc.base.bias.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, input_lengths):
        x = x.permute(0, 2, 1)
        x = self.cnn_stem(x)
        x = self.cnn_tail(x)
        x = x.permute(0, 2, 1)

        output_lengths = (input_lengths // 4).long().clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, output_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        x = self.dropout(x)
        x = self.out_norm(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)

        return nn.functional.log_softmax(x, dim=-1), output_lengths


def load_ctc_model(checkpoint_path, device='cpu'):
    state_dict = torch.load(checkpoint_path, map_location=device)
    is_lora_checkpoint = any(k.startswith('fc.base.') or k.startswith('fc.lora_') for k in state_dict.keys())

    if is_lora_checkpoint:
        lora_rank = 0
        if 'fc.lora_a' in state_dict:
            lora_rank = state_dict['fc.lora_a'].shape[0]
        model = BrailleCTCModel(use_lora=True, lora_rank=max(lora_rank, 1)).to(device)
    else:
        model = BrailleCTCModel().to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model