import torch

def epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, amp=True):
    train = optimizer is not None
    model.train(train)
    total_loss = correct = total = 0
    with (torch.enable_grad() if train else torch.no_grad()):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train: optimizer.zero_grad(set_to_none=True)
            enabled = amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled):
                logits = model(x); loss = loss_fn(logits, y)
            if train:
                if enabled:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            n = y.size(0); total += n
            total_loss += loss.item() * n
            correct += (logits.argmax(1) == y).sum().item()
    return total_loss/total, 100*correct/total
