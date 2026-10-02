# Segment 3: The Standard Training Loop

*Applied Accelerated AI (NPTEL, IIT Guwahati), Week 4, Instructor: Dr. Satyajit Das*

## Learning Objectives

1. Write a complete PyTorch training loop from scratch
2. Select and configure optimisers (SGD with momentum, Adam, AdamW) and know when to use each
3. Match common loss functions (CrossEntropyLoss, MSELoss, BCEWithLogitsLoss) to the right task
4. Use learning rate schedulers (CosineAnnealingLR, OneCycleLR) to improve convergence
5. Implement a validation loop, track the best model, and save full-state checkpoints

---

## 1. The Training Loop Overview

Training is a **loop** of the same operations repeated until convergence (loss close to its minimum):

1. **Forward pass:** run input through the network (weights start random)
2. **Compute loss:** how far predictions are from targets
3. **Backward pass:** compute the gradient of the loss w.r.t. every parameter (via autograd)
4. **Step:** update weights and biases using the gradients
5. **Reset gradients** for the next pass

---

## 2. The Six Lines Every Training Loop Must Have

```python
for epoch in range(num_epochs):
    model.train()                          # 1. Set training mode
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()              # 2. Clear old gradients
        predictions = model(batch_x)       # 3. Forward pass
        loss = criterion(predictions, batch_y)  # 4. Compute loss
        loss.backward()                    # 5. Compute gradients
        optimizer.step()                   # 6. Update weights
```

### Why each step matters (common mistakes)

| Step | Why it matters |
|---|---|
| `zero_grad()` before `backward()` | Gradients **accumulate**; without clearing, step N uses gradients from steps 1..N combined |
| `model.train()` at the start of each epoch | Dropout and BatchNorm behave differently in train vs eval; forgetting gives misleading results |
| `loss.backward()` before `optimizer.step()` | The optimiser needs the gradients that backward just computed |
| Move data to device before the forward pass | Model on CUDA + data on CPU gives a runtime error |

> **Key point:** Missing or reordering any step produces *silently wrong* results.

---

## 3. Production-Grade Training Step (LLM-style)

```python
optimizer.zero_grad(set_to_none=True)      # faster than zero_grad()
with torch.cuda.amp.autocast():            # optional: mixed precision
    logits = model(x)                      # forward pass
    loss   = F.cross_entropy(logits, y)    # loss function
scaler.scale(loss).backward()              # backward through scaled loss
torch.nn.utils.clip_grad_norm_(            # gradient clipping
    model.parameters(), max_norm=1.0)
scaler.step(optimizer)                     # optimiser step
scaler.update()                            # update loss scale
```

- **Logits** = raw outputs of the final layer, before softmax or other activation
- **Autocast (AMP):** forward pass + loss computed in mixed precision; omit it if you don't want mixed precision
- **Gradient clipping:** standard in modern LLM pipelines to keep gradients from blowing up
- **Pattern:** AMP + clipping + scaler

---

## 4. Optimisers: How Weights Are Updated

Optimisers apply the update rule using the gradients. Their settings (lr, momentum, weight decay) are **hyperparameters**: user-chosen values that affect training performance.

### SGD (Stochastic Gradient Descent)

```python
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9,
    weight_decay=1e-4, nesterov=True)
```

- Update rule: `w = w - lr * (grad + momentum * prev_update)`
- `momentum=0.9`: carries 90% of the previous update, smoothing noisy gradients
- `weight_decay`: L2 regularisation
- **Best for:** CNNs on vision tasks (classification, segmentation, detection) with careful LR tuning. Example: ResNet on ImageNet = SGD + OneCycle

### Adam

```python
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3,
    betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
```

- Maintains **per-parameter adaptive learning rates** from gradient history
- `betas`: first/second moment decay rates; rarely changed
- Usually works well without much tuning
- **Best for:** NLP and generative models

### AdamW (the modern default)

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4,
    weight_decay=0.1,   # higher than Adam
    fused=True)         # faster on CUDA
```

- **Difference from Adam:** weight decay is applied **directly to the weights**, not added to the gradient (decoupled weight decay, the mathematically correct form of L2 regularisation)
- **Use by default for:** LLMs, ViT, BERT, diffusion models, almost everything modern
- Typical LR: `1e-4` to `3e-4`; weight decay: `0.01` to `0.1`
- `fused=True` accelerates training on CUDA

### Quick comparison

| Optimiser | Typical use | Starting hyperparameters |
|---|---|---|
| SGD + momentum | CNNs / vision | lr=0.01, momentum=0.9, wd=1e-4 |
| Adam | NLP, generative | lr=1e-3, betas=(0.9, 0.999) |
| AdamW | LLMs, ViT, BERT, diffusion | lr=3e-4, wd=0.1 |

---

## 5. Learning Rate Scheduling

The learning rate is the primary hyperparameter controlling learning speed. A **higher LR is not always faster to converge**.

- **Static LR:** too low means slow training; too high means unstable
- **Adaptive/scheduled LR:** standard in modern training, since you don't want the same speed at every stage

**Intuition across training phases:**
- **Early:** learn coarse features, so go faster (warm-up)
- **Middle:** steadily decay
- **End:** learn fine details, so use a small LR

```python
# Cosine annealing: smoothly decays LR to 0
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs)

# OneCycleLR: warm up, then cosine decay
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4, total_steps=len(loader)*epochs)

scheduler.step()   # OneCycle: call after each *step*, not each epoch
```

- **Linear warm-up + cosine decay** is the standard for Transformer training, and also works for CNNs and diffusion models

---

## 6. Loss Functions: Matching Task to Criterion

### Multi-class classification: `CrossEntropyLoss`

```python
criterion = nn.CrossEntropyLoss()
logits = model(x)                    # (B, num_classes), raw scores
loss = criterion(logits, labels)     # labels: (B,) integers
```

- Takes **raw logits** (not softmax output); applies `log_softmax` internally
- `label_smoothing=0.1`: softens hard labels, reduces overconfidence
- `weight`: up-weights rare classes for imbalanced datasets

### Binary classification: `BCEWithLogitsLoss`

```python
criterion = nn.BCEWithLogitsLoss()
logits = model(x).squeeze()          # (B,)
labels = labels.float()              # must be float
loss = criterion(logits, labels)
```

- Numerically stable: combines sigmoid + BCE in one operation
- Use for a single probability output (yes/no, real/fake)

### Regression

```python
nn.MSELoss()     # mean squared error
nn.L1Loss()      # mean absolute error
nn.HuberLoss()   # smooth L1, robust to outliers
```

- **MSE:** penalises large errors heavily; sensitive to outliers
- **L1 (MAE):** more robust to outliers; non-differentiable at 0
- **Huber:** L2 near zero, L1 for large errors (best of both)

### Sequence modelling: CrossEntropy with `ignore_index`

```python
criterion = nn.CrossEntropyLoss(ignore_index=0)  # padding tokens ignored
logits = model(input_ids)                        # (B, T, vocab_size)
loss = criterion(logits.view(-1, V), targets.view(-1))
```

### Custom losses

```python
def focal_loss(logits, targets, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none')
    weight = (1 - p) ** gamma
    return (weight * ce).mean()
```

- Any differentiable function of tensors works as a loss; autograd handles backprop
- Built-in losses are a convenience, not a requirement

### Selection guide

| Task | Loss |
|---|---|
| Multi-class classification | `CrossEntropyLoss` |
| Binary classification | `BCEWithLogitsLoss` |
| Regression | `MSELoss` or `HuberLoss` |
| Language modelling | `CrossEntropyLoss` + `ignore_index` |
| Contrastive (CLIP) | InfoNCE / cosine similarity |
| Object detection | FocalLoss + GIoU + L1 |

---

## 7. Validation Loop

```python
def evaluate(model, loader, criterion, device):
    model.eval()                     # CRITICAL: disables Dropout/BatchNorm train behaviour
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():            # CRITICAL: no gradient computation needed
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += x.size(0)
    return total_loss / total, correct / total   # avg loss, accuracy
```

- Without `model.eval()` and `torch.no_grad()`: Dropout stays active and gradients waste memory, giving wrong metrics
- Keep the validation loop **separate** from the training loop

---

## 8. Checkpointing

### Save full training state

```python
checkpoint = {
    'epoch': epoch,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'scaler': scaler.state_dict(),   # AMP loss scale
    'val_loss': val_loss,
    'val_acc': val_acc,
}
torch.save(checkpoint, f'checkpoint_epoch{epoch:03d}.pth')
```

### Resume from checkpoint

```python
ckpt = torch.load('checkpoint_epoch005.pth', map_location=device)
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
scheduler.load_state_dict(ckpt['scheduler'])
start_epoch = ckpt['epoch'] + 1
```

> Note: if you use AMP, also restore the scaler: `scaler.load_state_dict(ckpt['scaler'])`. The slide omits this line.

### Track the best model

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), 'best_model.pth')
```

- Save a separate checkpoint whenever validation loss improves
- Load the best checkpoint at the end for final evaluation and deployment

---

## Summary

- **Six steps, in order:** `model.train()` → `zero_grad()` → forward → loss → `backward()` → `step()`
- **AdamW** is the default for modern deep learning (start with `lr=3e-4`, `weight_decay=0.1`)
- **Match loss to task:** CE for multi-class, BCEWithLogits for binary, MSE/Huber for regression, CE + `ignore_index` for language modelling
- **Validation needs** `model.eval()` and `torch.no_grad()`
- **Checkpoint the full state** (model, optimizer, scheduler, scaler, epoch), not just the weights, for exact resumption

**Next segment:** Data loading and performance optimisation.

---

## Notes on the Source Material

- The Adam slide uses `weight_decay=1e-2`, which is unusually high for plain Adam. That value is more typical of AdamW.
- The resume snippet doesn't restore the scaler state even though it is saved.
- The lecture says model.eval "switches off" Dropout and BatchNorm. Precisely: Dropout is disabled, and BatchNorm switches to using its running statistics instead of batch statistics.
