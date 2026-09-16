import matplotlib.pyplot as plt

# -----------------------
# Stage 1 data
# -----------------------
epochs_stage1 = [1, 2, 3, 4, 5, 6, 7, 8]
loss_stage1 = [1090.22, 260.77, 233.96, 206.29, 180.30, 164.44, 149.46, 139.36]

# -----------------------
# Stage 2 data
# -----------------------
epochs_stage2 = list(range(1, 16))
loss_stage2 = [
    56.5006, 26.2078, 18.5771, 14.9933, 12.6677,
    11.1948, 10.2135, 8.0239, 7.3087, 5.7575,
    4.9649, 4.9813, 4.8314, 3.7466, 3.0813
]

# =======================
# Plot Stage 1
# =======================
plt.figure(figsize=(8, 5))

plt.plot(epochs_stage1, loss_stage1, marker='o', linewidth=2, label='Stage 1 Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Stage 1 Training Loss Curve (Keypoint Detection)')
plt.legend()
plt.grid(True)

plt.savefig('stage1_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()


# =======================
# Plot Stage 2
# =======================
plt.figure(figsize=(8, 5))

plt.plot(epochs_stage2, loss_stage2, marker='s', linewidth=2, label='Stage 2 Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Stage 2 Training Loss Curve (Keypoint Detection)')
plt.legend()
plt.grid(True)

plt.savefig('stage2_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()