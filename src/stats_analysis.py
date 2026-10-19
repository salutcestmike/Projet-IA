import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

names = ["segmented_8bits_v1",
         "segmented_8bits_v2",
         "segmented_8bits_v2_downsampling2",
         "segmented_8bits_v2_downsampling4"]

dice_scores = {}

# Tracer les histogrammes sur le même graphique
plt.figure(figsize=(9, 5))

for name in names:
    df = pd.read_csv(f"outputs/stats/{name}.csv", index_col=0)
    dice_scores[name] = df["Dice Score"].dropna()

    print(f"\n{name}")
    print(f"Nombre de scores : {len(dice_scores[name])}")
    print(f"Moyenne          : {dice_scores[name].mean():.4f}")
    print(f"Écart-type       : {dice_scores[name].std():.4f}")
    print(f"Médiane          : {dice_scores[name].median():.4f}")


    plt.hist(dice_scores[name], bins=30, alpha=0.6, label=name)

plt.xlabel("Score Dice")
plt.ylabel("Nombre d'images")
plt.title("Distribution des scores Dice")
plt.xlim(0, 1)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()