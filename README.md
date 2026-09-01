## Results

Static gestures (5 classes, 300 samples, session-stratified split):

| Model | Params | Best val acc | Epochs to 100% |
|-------|--------|--------------|----------------|
| Transformer | 73,413 | 1.000 | 2 |
| MLP (mean-pooled) | 25,349 | 1.000 | 27 |

Attention gives no accuracy benefit on static poses — the mean-pooled baseline
matches it with 2.9x fewer parameters. Next: dynamic gestures, where temporal
structure should matter.