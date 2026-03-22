import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset/complete_vector.csv')
imbalance_counts = df['Label'].value_counts()
imbalance_percentages = df['Label'].value_counts(normalize=True) * 100

print("--- Class Imbalance Report ---")
print(f"Typical (0): {imbalance_counts[0]} students ({imbalance_percentages[0]:.2f}%)")
print(f"At-Risk (1): {imbalance_counts[1]} students ({imbalance_percentages[1]:.2f}%)")

plt.figure(figsize=(8, 6))
bars = plt.bar(['Typical (0)', 'At-Risk (1)'], imbalance_counts, color=['#4CAF50', '#F44336'])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, int(yval), ha='center', va='bottom', fontweight='bold')

plt.title('Distribution of Students: Typical vs. At-Risk')
plt.ylabel('Number of Students')
plt.show()