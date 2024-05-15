import matplotlib.pyplot as plt
import math
import pandas as pd

# =======================================================================================
# Perform basic exploratory analysis
# =======================================================================================

# -------------------------------------------------------------------------------
# Load test dataset
# -------------------------------------------------------------------------------

selected_spotify_data = pd.read_csv('../../1_data/derived/selected_spotify_data.csv')

# -------------------------------------------------------------------------------
# Plot histogram of each feature
# -------------------------------------------------------------------------------

# Select all columns except the last two (genre, music category)
columns_to_include = selected_spotify_data.iloc[:, 0:-2]

# Determine the number of rows and columns for the subplot grid
num_columns = columns_to_include.shape[1]
num_rows = math.ceil(num_columns / 3)

# Create a ...x3 grid of subplots
fig, axes = plt.subplots(num_rows, 3, figsize=(16, 10))

# Flatten the axes array to facilitate iteration
axes = axes.flatten()

# Plot histograms for each column
for i, column in enumerate(columns_to_include.columns):
    ax = axes[i]  # Get the current axis
    ax.hist(columns_to_include[column], bins=20, color='skyblue', edgecolor='black', zorder=2)
    ax.set_title(column)  # Set the title of the subplot
    ax.set_xlabel('Value')  # Set the x-axis label
    ax.set_ylabel('Frequency')  # Set the y-axis label
    ax.grid(True, zorder=0)  # Add gridlines

# Hide empty subplots if necessary
for i in range(num_columns, num_rows * 3):
    fig.delaxes(axes[i])

# Adjust layout
plt.tight_layout()

# Save the plot as an image file with high resolution (300 dpi)
plt.savefig("histograms_of_features.png", dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

# -------------------------------------------------------------------------------
# Plot music categories counts bar plot
# -------------------------------------------------------------------------------

# Count the frequencies of each category
category_counts = selected_spotify_data['music_category'].value_counts()

# Create a bar plot
plt.figure(figsize=(12, 8))
category_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Music Category')
plt.ylabel('Frequency')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()

# Save the plot as an image file with high resolution (300 dpi)
plt.savefig("music_categories_counts.png", dpi=300, bbox_inches='tight')

# Display the plot
plt.show()