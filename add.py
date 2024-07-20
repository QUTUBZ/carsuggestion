import pandas as pd
import numpy as np

# Load your original dataset
df = pd.read_csv('car_dataset.csv')

# Generate engine values and round them
engine_values = np.arange(1, 2.7, 0.2)
engine_values_rounded = np.round(engine_values, 1)

# Shuffle the rounded engine values
np.random.shuffle(engine_values_rounded)

# Assign to the 'engine' column
df['engine'] = np.random.choice(engine_values_rounded, size=len(df))

# Define km driven intervals and assign evenly
intervals = [(20000, 40000), (40000, 60000), (60000, 80000), (80000, 100000), (100000, 130000)]

# Calculate number of cars per interval
cars_per_interval = len(df) // len(intervals)

# List to hold assigned indices
assigned_indices = []

# Assign km driven values within each interval
for idx, (start, end) in enumerate(intervals):
    # Randomly select indices for this interval
    indices = np.random.choice(df.index.difference(assigned_indices), size=cars_per_interval, replace=False)
    assigned_indices.extend(indices)
    
    # Assign km driven values within the interval
    df.loc[indices, 'km_driven'] = np.random.randint(start, end, size=len(indices))

# For any remaining cars, assign to the last interval
remaining_indices = df.index.difference(assigned_indices)
df.loc[remaining_indices, 'km_driven'] = np.random.randint(intervals[-1][0], intervals[-1][1], size=len(remaining_indices))

# Create a new DataFrame with all columns
cardemo_df = df.copy()

# Save as cardemo.csv
cardemo_df.to_csv('cardemo.csv', index=False)

