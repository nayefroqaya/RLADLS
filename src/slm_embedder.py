from pathlib import Path
from typing import Dict
import hashlib
import pickle

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    summed = torch.sum(
        token_embeddings * input_mask_expanded,
        dim=1
    )

    counts = torch.clamp(
        input_mask_expanded.sum(dim=1),
        min=1e-9
    )

    return summed / counts


def make_template_fingerprint(
    template_to_id: Dict[str, int],
    model_name: str
) -> str:
    items = sorted(
        template_to_id.items(),
        key=lambda x: x[1]
    )

    raw = model_name + "\n" + "\n".join(
        [f"{template_id}:{template}" for template, template_id in items]
    )

    return hashlib.md5(
        raw.encode("utf-8")
    ).hexdigest()


def build_minilm_embedding_matrix(
    template_to_id: Dict[str, int],
    model_name: str,
    cache_dir: str,
    batch_size: int,
    device: torch.device
) -> torch.Tensor:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    fingerprint = make_template_fingerprint(
        template_to_id=template_to_id,
        model_name=model_name
    )

    safe_model_name = model_name.replace("/", "_")
    cache_file = cache_path / f"{safe_model_name}_{fingerprint}.pkl"

    if cache_file.exists():
        print(f"Loading cached MiniLM embeddings: {cache_file}")

        with open(cache_file, "rb") as file:
            embedding_matrix = pickle.load(file)

        return torch.tensor(
            embedding_matrix,
            dtype=torch.float32
        )

    print(f"Building MiniLM embeddings using: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.to(device)
    model.eval()

    templates_sorted = sorted(
        template_to_id.items(),
        key=lambda x: x[1]
    )

    template_texts = [
        template
        for template, _ in templates_sorted
    ]

    template_ids = [
        template_id
        for _, template_id in templates_sorted
    ]

    all_embeddings = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(template_texts), batch_size),
            desc="MiniLM template embedding"
        ):
            batch_texts = template_texts[start:start + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            output = model(**encoded)

            embeddings = mean_pooling(
                model_output=output,
                attention_mask=encoded["attention_mask"]
            )

            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1
            )

            all_embeddings.append(
                embeddings.cpu().numpy()
            )

    template_embeddings = np.vstack(all_embeddings)
    slm_dim = template_embeddings.shape[1]
    vocab_size = max(template_to_id.values()) + 1

    embedding_matrix = np.zeros(
        (vocab_size, slm_dim),
        dtype=np.float32
    )

    for row_index, template_id in enumerate(template_ids):
        embedding_matrix[template_id] = template_embeddings[row_index]

    with open(cache_file, "wb") as file:
        pickle.dump(embedding_matrix, file)

    print(f"Saved MiniLM embeddings to: {cache_file}")

    return torch.tensor(
        embedding_matrix,
        dtype=torch.float32
    )
