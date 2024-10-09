import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Excel file , change the excel file name!
df = pd.read_excel('INWH_AI.xlsx')

# Custom ranking function to ensure suppliers with "0" amount are ranked last
def custom_rank(group):
    # Rank suppliers with non-zero offers
    group['Rank'] = group.loc[group['Offer'] > 0, 'Offer'].rank(method='min', ascending=True)
    # Assign the highest rank to suppliers with "0" amount
    group.loc[group['Offer'] == 0, 'Rank'] = group['Rank'].max() + 1
    return group

# Apply the custom ranking function to each Location group
df = df.groupby('Location').apply(custom_rank).reset_index(drop=True)

# Calculate the number of suppliers to be eliminated per Location
elimination_count = df.groupby('Location')['Supplier'].transform(lambda x: np.ceil(len(x) / 2))

# Label the top 50% suppliers with higher offers as 'eliminated'
df['Status'] = np.where(df['Rank'] <= elimination_count, 'Active', 'Eliminated')

# Sort the DataFrame based on 'Location' and then 'Rank'
df_sorted = df.sort_values(by=['Location', 'Rank'])

# Write the sorted DataFrame with status labels to a new Excel file, do not forget to change the file name!
df_sorted.to_excel('R2_[Warehouse]_eval_LSPs_per_Location09102024.xlsx', index=False)

print('Suppliers have been evaluated, ranked, and the results are saved to "R2_[Warehouse]_eval_LSPs_per_Location09102024.xlsx".')

# Create a box plot to show outliers per Location
plt.figure(figsize=(14, 10))
sns.boxplot(x='Location', y='Offer', data=df)
plt.title('Box Plot of Offers per Location', fontsize=16)
plt.xlabel('Location', fontsize=14)
plt.ylabel('Offer', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Calculate and annotate outliers
for location in df['Location'].unique():
    location_data = df[df['Location'] == location]
    q1 = location_data['Offer'].quantile(0.25)
    q3 = location_data['Offer'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = location_data[(location_data['Offer'] < lower_bound) | (location_data['Offer'] > upper_bound)]
    
    for _, row in outliers.iterrows():
        plt.annotate(row['Supplier'], (row['Location'], row['Offer']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='red', arrowprops=dict(arrowstyle='->', color='red'))

# Save and show the plot
plt.tight_layout()
plt.savefig('Box_Plot_Outliers_Per_Location.png')
plt.show()