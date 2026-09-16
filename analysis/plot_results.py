import pandas as pd
import matplotlib.pyplot as plt

# IMPORTANT: change path if needed
df = pd.read_csv("D:/tennis_project/analysis/training/runs/detect/train3/results.csv")

print(df.columns)

# LOSS CURVE
plt.figure()
plt.plot(df['epoch'], df['train/box_loss'], label='Train Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid()

plt.savefig("loss_curve.png")
plt.show()