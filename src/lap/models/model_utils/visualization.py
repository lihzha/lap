def log_attention_mask_wandb(mask, name: str = "attention_mask") -> None:
    """Log a boolean/0-1 attention mask image to Weights & Biases."""
    import matplotlib.pyplot as plt
    import numpy as np
    import wandb

    mask = np.asarray(mask, dtype=float)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("Key positions")
    ax.set_ylabel("Query positions")
    ax.set_title(name)

    wandb.log({name: wandb.Image(fig)})
    plt.close(fig)
