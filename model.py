import torch
import torch.nn as nn
from config import AMINO_ACID_VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES, MAX_LEN

class CNNSecondaryStructureV2(nn.Module):
    def __init__(self):
        super(CNNSecondaryStructureV2, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=AMINO_ACID_VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=0
        )

        self.positional_embedding = nn.Embedding(MAX_LEN, EMBEDDING_DIM)

        # Progressive conv layers
        self.conv1 = nn.Conv1d(EMBEDDING_DIM, 128, kernel_size=3, padding=1)  # fine detail
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)  # mid-level motif
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=11, padding=5)  # full helix/sheet
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)

        self.classifier = nn.Conv1d(64, NUM_CLASSES, kernel_size=1)  # 1x1 conv

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = self.embedding(x) + self.positional_embedding(positions)  # [B, L, E]

        x = x.permute(0, 2, 1)  # [B, E, L]
        x = self.dropout1(self.relu1(self.conv1(x)))
        x = self.dropout2(self.relu2(self.conv2(x)))
        x = self.dropout3(self.relu3(self.conv3(x)))

        x = self.classifier(x)  # [B, NUM_CLASSES, L]
        x = x.permute(0, 2, 1)  # [B, L, NUM_CLASSES]
        return x
