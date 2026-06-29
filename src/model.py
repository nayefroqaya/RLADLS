import torch
import torch.nn as nn


class SLM_DQN(nn.Module):
    """
    DQN model using:
      template-ID embedding + frozen MiniLM semantic embedding + GRU.
    """

    def __init__(
        self,
        vocab_size: int,
        id_embedding_dim: int,
        hidden_dim: int,
        slm_embedding_matrix: torch.Tensor,
        num_actions: int = 2,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.id_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=id_embedding_dim,
            padding_idx=padding_idx
        )

        self.slm_embedding = nn.Embedding.from_pretrained(
            embeddings=slm_embedding_matrix.float(),
            freeze=True,
            padding_idx=padding_idx
        )

        slm_embedding_dim = slm_embedding_matrix.shape[1]
        # for full sytsem
        #combined_dim = id_embedding_dim + slm_embedding_dim

        # for ablation :
        combined_dim = id_embedding_dim

        self.gru = nn.GRU(
            input_size=combined_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    # full system
    #def forward(self, x):
    #    id_emb = self.id_embedding(x)
    #    slm_emb = self.slm_embedding(x)

    #    combined = torch.cat(
    #        [id_emb, slm_emb],
    #        dim=-1
    #    )

    #for ablation :
    # Template-ID only ablation
    def forward(self, x):
        id_emb = self.id_embedding(x)

        _, hidden = self.gru(id_emb)
        hidden = hidden[-1]

        return self.head(hidden)



        _, hidden = self.gru(combined)
        hidden = hidden[-1]

        return self.head(hidden)
