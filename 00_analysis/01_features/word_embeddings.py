from transformers import BertTokenizer,BertModel
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')
X = df["comment_text"].tolist()
y = df["toxic"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

def bert_mean_pool_embeddings(
    texts,
    model_name="bert-base-uncased",
    batch_size=32,
    max_length=128,
    device=None,
    return_numpy=True
):
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model + tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling with attention mask
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

        mean_embeddings = sum_embeddings / sum_mask

        # Move to CPU
        mean_embeddings = mean_embeddings.cpu()

        if return_numpy:
            mean_embeddings = mean_embeddings.numpy()

        all_embeddings.append(mean_embeddings)

    # Combine all batches
    if return_numpy:
        import numpy as np
        return np.vstack(all_embeddings)
    else:
        return torch.cat(all_embeddings, dim=0)


texts = X_train.copy()
embeddings = bert_mean_pool_embeddings(
    texts,
    batch_size=32
)