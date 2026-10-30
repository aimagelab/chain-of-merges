import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
    pos = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return (attention_mask.bool() * pos).argmax(dim=1)


def nli_logits(model, head: torch.Tensor, task_data, batch, device) -> torch.Tensor:
    ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
    hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    pooled = hidden[torch.arange(ids.shape[0], device=device), last_token_index(mask)]
    logits = F.linear(pooled.float(), head.to(device).float())
    if task_data.masked_class is not None:
        logits = logits.index_fill(1, torch.tensor([task_data.masked_class], device=device), float("-inf"))
    return logits


@torch.no_grad()
def evaluate_nli(model, head, task_data, batch_size: int = 32, device="cpu", split: str = "test") -> float:
    model.eval().to(device)
    loader = DataLoader(task_data.split(split), batch_size=batch_size, shuffle=False, collate_fn=task_data.collate_fn)
    correct = total = 0
    for batch in loader:
        ids, mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
        hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        pooled = hidden[torch.arange(ids.shape[0], device=device), last_token_index(mask)]
        logits = F.linear(pooled, head.to(device=device, dtype=pooled.dtype)).float()
        if task_data.masked_class is not None:
            logits[:, task_data.masked_class] = float("-inf")
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)
