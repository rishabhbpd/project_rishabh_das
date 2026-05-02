# ── Image dimensions ──────────────────────────────────────────────────
resize_x        = 224
resize_y        = 224
input_channels  = 3

# ── Training hyperparameters ───────────────────────────────────────────
batch_size      = 16
epochs          = 10
learning_rate   = 1e-3
dropout         = 0.5

# ── Dataset ───────────────────────────────────────────────────────────
subset          = 1000
val_subset      = 200

# ── Paths ─────────────────────────────────────────────────────────────
checkpoint_path = "checkpoints/final_weights.pth"
