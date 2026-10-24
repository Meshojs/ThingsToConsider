## Things to Consider When Training a Transformer

### 1. Mixed Precision
- Use FP16 or BF16 to save GPU memory and speed up training.
- Choose based on what your GPU supports.

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    y_pred = model(x)
    loss = criterion(y_pred, y)
```

### 2. Learning Rate Scheduler
- Use warmup at the beginning of training.
- Decay the learning rate as training progresses.

```python
optimizer.step()
scheduler.step()
```

### 3. Weight Decay
- Weight decay is a form of regularization.
- Usually, we apply it to weight matrices, not biases or normalization parameters.

```python
optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": 0.1},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=3e-4)
```

### 4. Reset Gradients
- PyTorch accumulates gradients, so reset them before each training step.

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
