import pandas as pd

# Load the CSV
df = pd.read_csv("coarse_grid_results.csv")

# Calculate the score (FP have greater impact on lowering score than FN)
W_fp = 1  # weight for false positives
W_fn = 1 # weight for false negatives

df["score"] = 100 - (((W_fp * df["FP"]) + (W_fn * df["FN"])) / (676)) * 100

# Sort by score descending
df_sorted = df.sort_values(by="score", ascending=False).reset_index(drop=True)

# Add Rank column (starting at 1)
df_sorted.insert(0, "Rank", df_sorted.index + 1)

# Select and reorder desired columns
final_df = df_sorted[["Rank", "LV", "Sigma", "BinaryThresh", "CircThresh", "score", "FP", "FN"]]

# Save to new CSV
final_df.to_csv("ranked_scores.csv", index=False)

print("Ranking complete. Output saved to 'ranked_scores.csv'")
