import pandas as pd
import numpy as np

# --- 1. Setup ---
num_rows = 2000
possible_genders = ["Male", "Female", "Non-Binary"]
filename = "realistic_body_data.csv"

# --- 2. Create Base Data ---
# We'll build the DataFrame column by column to create relationships.

# Start with independent variables: gender and age
df = pd.DataFrame({
    'gender': np.random.choice(possible_genders, size=num_rows, p=[0.48, 0.48, 0.04]),
    'age': np.random.randint(18, 66, size=num_rows)
})

# --- 3. Generate Correlated Measurements ---
# We'll use normal distributions (np.random.normal) for more realistic "bell curve" data
# and add "noise" to make it less perfect.

# Create masks to apply gender-specific logic
male_mask = (df['gender'] == 'Male')
female_mask = (df['gender'] == 'Female')
nb_mask = (df['gender'] == 'Non-Binary')

# --- Height (cm) ---
# Set different average heights (mean) and spreads (std dev) for each gender
df.loc[male_mask, 'height'] = np.random.normal(loc=175.3, scale=7, size=male_mask.sum())
df.loc[female_mask, 'height'] = np.random.normal(loc=161.8, scale=6, size=female_mask.sum())
df.loc[nb_mask, 'height'] = np.random.normal(loc=168.5, scale=8, size=nb_mask.sum()) # Broader mix

# --- Weight (kg) ---
# Weight is heavily correlated with height (using a rough BMI calculation) + some noise
# Base BMI calculation: np.random.normal(loc=24, scale=4)
df['weight'] = (df['height'] / 100)**2 * np.random.normal(loc=24, scale=4, size=num_rows) + \
               np.random.normal(loc=0, scale=3, size=num_rows) # Extra noise

# --- Waist, Hips, Chest (cm) ---
# These are correlated with height, weight, and gender.
# The formulas are estimations to create realistic trends.

# Waist
df['waist'] = 20 + (df['weight'] * 0.7) + (df['height'] * 0.1) + \
              np.random.normal(loc=0, scale=4, size=num_rows)
df.loc[male_mask, 'waist'] += np.random.normal(loc=5, scale=2, size=male_mask.sum()) # Males tend to have larger waists relative to weight/height

# Hips
df['hips'] = 25 + (df['weight'] * 0.6) + (df['height'] * 0.2) + \
             np.random.normal(loc=0, scale=4, size=num_rows)
df.loc[female_mask, 'hips'] += np.random.normal(loc=8, scale=3, size=female_mask.sum()) # Females tend to have larger hips

# Chest
df['chest'] = 40 + (df['weight'] * 0.4) + (df['height'] * 0.15) + \
              np.random.normal(loc=0, scale=5, size=num_rows)
df.loc[male_mask, 'chest'] += np.random.normal(loc=7, scale=3, size=male_mask.sum()) # Males tend to have broader chests

# --- 4. Clean Up and Add Noise ---

# Round the data to make it look more natural
df['height'] = df['height'].round(1)
df['weight'] = df['weight'].round(1)
df[['waist', 'chest', 'hips']] = df[['waist', 'chest', 'hips']].round(0).astype(int)

# Introduce missing data (NaNs) to simulate real-world "noise"
cols_to_corrupt = ['height', 'weight', 'waist', 'chest', 'hips']
for col in cols_to_corrupt:
    # Randomly select ~3% of indices to set to NaN
    nan_indices = df.sample(frac=0.03).index
    df.loc[nan_indices, col] = np.nan

# --- 5. Finalize and Save ---

# Ensure column order matches your header
header = ["age", "gender", "height", "weight", "waist", "chest", "hips"]
df = df[header]

# Save to CSV
df.to_csv(filename, index=False)

print(f"Successfully created '{filename}' with {num_rows} realistic random rows.")
print("\nFirst 5 rows of the dataset:")
print(df.head())
