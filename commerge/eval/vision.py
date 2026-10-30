import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_vision(model, head: torch.Tensor, dataset, device, batch_size: int = 128, num_workers: int = 8) -> float:
    model.eval().to(device)
    head = head.to(device)
    correct = total = 0
    for pixels, labels in tqdm(DataLoader(dataset, batch_size=batch_size, num_workers=num_workers), "eval", leave=False):
        emb = model(pixels.to(device))
        logits = (emb / emb.norm(dim=-1, keepdim=True)) @ head.T
        correct += (logits.argmax(-1).cpu() == labels).sum().item()
        total += len(labels)
    return correct / total
