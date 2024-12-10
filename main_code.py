import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import re
import textwrap
import numpy as np

file_path_updated = r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\portfolio_DS\postdocs\unis.csv"
data_updated = pd.read_csv(file_path_updated, delimiter=';', encoding='utf-8')

def extract_coordinates(df, col_name):
    df[f'{col_name}_degree'] = df[col_name].str.extract(r'(\d+\.\d+)')
    df[f'{col_name}_direction'] = df[col_name].str.extract(r'°\s*([NSWE])')
    df[col_name] = pd.to_numeric(df[f'{col_name}_degree'], errors='coerce')
    df.loc[df[f'{col_name}_direction'].isin(['S', 'W']), col_name] *= -1
    df.drop(columns=[f'{col_name}_degree', f'{col_name}_direction'], inplace=True)

extract_coordinates(data_updated, 'Latitude')
extract_coordinates(data_updated, 'Longitude')

data_updated.dropna(subset=['Latitude', 'Longitude'], inplace=True)

def split_numbers(s):
    if pd.isna(s):
        return []
    return re.findall(r'-?\d+\.\d+', s)

data_updated['Latitude_list'] = data_updated['Latitude'].astype(str).apply(split_numbers)
data_updated['Longitude_list'] = data_updated['Longitude'].astype(str).apply(split_numbers)
data_updated = data_updated.explode('Latitude_list')
data_updated = data_updated.explode('Longitude_list')
data_updated['Latitude'] = data_updated['Latitude_list'].astype(float)
data_updated['Longitude'] = data_updated['Longitude_list'].astype(float)
data_updated.drop(columns=['Latitude_list', 'Longitude_list'], inplace=True)
data_updated[['Latitude', 'Longitude']].head()

# Calculate salary-to-cost-of-living ratios
data_updated['ratio_min'] = data_updated['min_salary'] / data_updated['cost_of_living_max']
data_updated['ratio_max'] = data_updated['max_salary'] / data_updated['cost_of_living_min']
data_updated['ratio_avg'] = (data_updated['ratio_min'] + data_updated['ratio_max']) / 2

# Identify the top 5 best and worst ratios
top_5_best = data_updated.nlargest(5, 'ratio_avg')
top_5_worst = data_updated.nsmallest(5, 'ratio_avg')
top_5_combined = pd.concat([top_5_worst, top_5_best])
top_5_combined['University'] = top_5_combined['University'].apply(lambda x: '\n'.join(textwrap.wrap(x, 20)))

# Create a bar plot for the top 5 best and worst ratios
fig, ax = plt.subplots(figsize=(14, 10))  
top_5_combined_sorted = top_5_combined.sort_values('ratio_avg')  
top_5_combined_sorted.plot(
    kind='barh',
    x='University',
    y='ratio_avg',
    ax=ax,
    color=['red'] * 5 + ['green'] * 5,
    legend=False
)

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=10, padding=10)  # Increased padding

handles = [plt.Line2D([0], [0], color='red', lw=4),
           plt.Line2D([0], [0], color='green', lw=4)]
labels = ['Worst Ratios', 'Best Ratios']
ax.legend(handles, labels, loc='lower right', fontsize=12)

ax.set_xlabel('')
ax.set_ylabel('')

ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig("top_5_best_worst_ratios.png", dpi=300, bbox_inches="tight")
plt.show()

# Load the Natural Earth dataset
shapefile_path = r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\portfolio_DS\postdocs\ne_110m_admin_0_countries.shp"
world = gpd.read_file(shapefile_path)

europe = world[world['CONTINENT'] == 'Europe']

geometry = [Point(lon, lat) for lon, lat in zip(data_updated['Longitude'], data_updated['Latitude'])]
data_gdf = gpd.GeoDataFrame(data_updated, geometry=geometry, crs="EPSG:4326")

fig, ax = plt.subplots(1, 1, figsize=(16, 12))

europe.boundary.plot(ax=ax, color="black", linewidth=0.5)
europe.plot(ax=ax, color="#f0f0f0", edgecolor="black", linewidth=0.5)

points = data_gdf.plot(
    ax=ax,
    column='ratio_avg',
    cmap='coolwarm_r',  
    markersize=100,
    legend=True,
    legend_kwds={
        'label': "Postdoc Salary-to-Cost-of-Living Ratio",
        'orientation': "vertical",
        'shrink': 0.8
    }
)

ax.set_xlim(-30, 60) 
ax.set_ylim(30, 72)   
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
scalebar = plt.Line2D([0], [0], color='black', linewidth=2, label='Scalebar (Optional)')
ax.legend([scalebar], ["Approx. Scale"], loc='lower left', fontsize=10)

plt.savefig("postdoc_static_map_professional.png", dpi=300, bbox_inches="tight")
plt.show()

##add life satistication data in the ratio
data_updated['life_satisfaction'] = pd.to_numeric(data_updated['life_satisfaction'], errors='coerce')
data_updated['min_salary'] = pd.to_numeric(data_updated['min_salary'], errors='coerce')
data_updated['max_salary'] = pd.to_numeric(data_updated['max_salary'], errors='coerce')
data_updated['cost_of_living_min'] = pd.to_numeric(data_updated['cost_of_living_min'], errors='coerce')
data_updated['cost_of_living_max'] = pd.to_numeric(data_updated['cost_of_living_max'], errors='coerce')
data_updated['adjusted_ratio_min'] = data_updated['min_salary'] * data_updated['life_satisfaction'] / data_updated['cost_of_living_max']
data_updated['adjusted_ratio_max'] = data_updated['max_salary'] * data_updated['life_satisfaction'] / data_updated['cost_of_living_min']
data_updated['adjusted_ratio_avg'] = (data_updated['adjusted_ratio_min'] + data_updated['adjusted_ratio_max']) / 2

top_5_best_adjusted = data_updated.nlargest(5, 'adjusted_ratio_avg')
top_5_worst_adjusted = data_updated.nsmallest(5, 'adjusted_ratio_avg')
top_5_combined_adjusted = pd.concat([top_5_worst_adjusted, top_5_best_adjusted])
top_5_combined_adjusted['University'] = top_5_combined_adjusted['University'].apply(lambda x: '\n'.join(textwrap.wrap(x, 20)))

fig, ax = plt.subplots(figsize=(14, 10))  
top_5_combined_adjusted_sorted = top_5_combined_adjusted.sort_values('adjusted_ratio_avg')  # Sort to have worst at bottom
top_5_combined_adjusted_sorted.plot(
    kind='barh',
    x='University',
    y='adjusted_ratio_avg',
    ax=ax,
    color=['red'] * 5 + ['green'] * 5,
    legend=False
)

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=10, padding=10)  
handles = [plt.Line2D([0], [0], color='red', lw=4),
           plt.Line2D([0], [0], color='green', lw=4)]
labels = ['Worst Adjusted Ratios', 'Best Adjusted Ratios']
ax.legend(handles, labels, loc='lower right', fontsize=12)
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig("top_5_best_worst_adjusted_ratios.png", dpi=300, bbox_inches="tight")
plt.show()


# Load the QS rankings data
data_new= data_updated
qs_file_path = r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\portfolio_DS\postdocs\2025 QS World University Rankings 2.2.xlsx"
qs_rankings = pd.read_excel(qs_file_path)

qs_cleaned = qs_rankings.iloc[3:, [3, 1, 4]] 
qs_cleaned.columns = ['University', 'QS_Rank', 'Location']
qs_cleaned['University'] = qs_cleaned['University'].str.strip()
qs_cleaned['QS_Rank'] = pd.to_numeric(qs_cleaned['QS_Rank'], errors='coerce')

# Fuzzy match and merge 
def fuzzy_match_university(university_name, qs_names, threshold=90):
    match = process.extractOne(university_name, qs_names, scorer=fuzz.token_set_ratio)
    if match and match[1] >= threshold:
        return match[0]
    return None

qs_university_names = qs_cleaned['University'].unique()
data_new['Fuzzy_Matched_University'] = data_new['University'].apply(
    lambda x: fuzzy_match_university(x, qs_university_names)
)

# Merge datasets
data_fuzzy_merged = pd.merge(
    data_new,
    qs_cleaned,
    left_on='Fuzzy_Matched_University',
    right_on='University',
    how='left',
    suffixes=('', '_QS')
)


output_file_path = 'merged_university_dataset_with_QS_rankings.csv'
data_fuzzy_merged.to_csv(output_file_path, index=False)
print(data_fuzzy_merged.head())

# Normalize QS Rank
qs_min = data_fuzzy_merged['QS_Rank'].min()
qs_max = data_fuzzy_merged['QS_Rank'].max()
data_fuzzy_merged['Normalized_QS_Rank'] = (qs_max - data_fuzzy_merged['QS_Rank']) / (qs_max - qs_min)

# Normalize Adjusted Ratio
adj_ratio_min = data_fuzzy_merged['adjusted_ratio_avg'].min()
adj_ratio_max = data_fuzzy_merged['adjusted_ratio_avg'].max()
data_fuzzy_merged['Normalized_Adjusted_Ratio'] = (
    data_fuzzy_merged['adjusted_ratio_avg'] - adj_ratio_min
) / (adj_ratio_max - adj_ratio_min)

# Recalculate Composite Index
data_fuzzy_merged['Composite_Index'] = (
    0.5 * data_fuzzy_merged['Normalized_QS_Rank'] +
    0.5 * data_fuzzy_merged['Normalized_Adjusted_Ratio']
)

data_fuzzy_merged_sorted = data_fuzzy_merged.sort_values(by='Composite_Index', ascending=False)

output_file_path = 'university_dataset_with_composite_index.csv'
data_fuzzy_merged_sorted.to_csv(output_file_path, index=False)

# Calculate Adjusted Ratios with Life Satisfaction in Numerator
data_fuzzy_merged['adjusted_ratio_min'] = (
    data_fuzzy_merged['min_salary'] * data_fuzzy_merged['life_satisfaction']
) / data_fuzzy_merged['cost_of_living_max']
data_fuzzy_merged['adjusted_ratio_max'] = (
    data_fuzzy_merged['max_salary'] * data_fuzzy_merged['life_satisfaction']
) / data_fuzzy_merged['cost_of_living_min']
data_fuzzy_merged['adjusted_ratio_avg'] = (
    data_fuzzy_merged['adjusted_ratio_min'] + data_fuzzy_merged['adjusted_ratio_max']
) / 2

#  Calculate Composite Index
data_fuzzy_merged['Inverted_QS_Rank'] = data_fuzzy_merged['QS_Rank'].max() - data_fuzzy_merged['QS_Rank'] + 1
data_fuzzy_merged['Normalized_QS_Rank'] = data_fuzzy_merged['Inverted_QS_Rank'] / data_fuzzy_merged['Inverted_QS_Rank'].max()
data_fuzzy_merged['Normalized_Adjusted_Ratio'] = data_fuzzy_merged['adjusted_ratio_avg'] / data_fuzzy_merged['adjusted_ratio_avg'].max()
data_fuzzy_merged['Composite_Index'] = 0.5 * data_fuzzy_merged['Normalized_QS_Rank'] + \
                                       0.5 * data_fuzzy_merged['Normalized_Adjusted_Ratio']

data_fuzzy_merged_sorted = data_fuzzy_merged.sort_values(by='Composite_Index', ascending=False)

# Display top 10 universities
top_universities = data_fuzzy_merged_sorted.head(10)
print(top_universities[['University', 'QS_Rank', 'adjusted_ratio_avg', 'Composite_Index']])

# Visualize Top Universities
top_universities = data_fuzzy_merged_sorted.head(10)
print(top_universities[['University', 'QS_Rank', 'adjusted_ratio_avg', 'Composite_Index']])

fig, ax = plt.subplots(figsize=(12, 8))
top_universities.plot(
    kind='barh',
    x='University',
    y='Composite_Index',
    ax=ax,
    color='skyblue',
    legend=False
)

ax.set_title('Top 10 Universities by Composite Index', fontsize=16)
ax.set_xlabel('Composite Index (Higher is Better)', fontsize=12)
ax.set_ylabel('', fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("top_10_universities_composite_index.png", dpi=300, bbox_inches="tight")
plt.show()

# Create a map for the Composite Index
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

europe.boundary.plot(ax=ax, color="black", linewidth=0.5)
europe.plot(ax=ax, color="#f0f0f0", edgecolor="black", linewidth=0.5)

points = data_gdf.plot(
    ax=ax,
    column='Composite_Index',
    cmap='viridis',  
    markersize=100,
    legend=True,
    legend_kwds={
        'label': "Composite Index",
        'orientation': "vertical",
        'shrink': 0.8
    }
)

ax.set_xlim(-30, 60) 
ax.set_ylim(30, 72)   
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
scalebar = plt.Line2D([0], [0], color='black', linewidth=2, label='Scalebar (Optional)')
ax.legend([scalebar], ["Approx. Scale"], loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig("postdoc_composite_index_map.png", dpi=300, bbox_inches="tight")
plt.show()

# Create a GeoDataFrame for Composite Index
geometry_composite = [Point(lon, lat) for lon, lat in zip(data_fuzzy_merged_sorted['Longitude'], data_fuzzy_merged_sorted['Latitude'])]
data_fuzzy_gdf_composite = gpd.GeoDataFrame(data_fuzzy_merged_sorted, geometry=geometry_composite, crs="EPSG:4326")

# Create a map for the Composite Index
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

europe.boundary.plot(ax=ax, color="black", linewidth=0.5)
europe.plot(ax=ax, color="#f0f0f0", edgecolor="black", linewidth=0.5)

data_fuzzy_gdf_composite.plot(
    ax=ax,
    column='Composite_Index',
    cmap='viridis',  
    markersize=100,
    legend=True,
    legend_kwds={
        'label': "Composite Index",
        'orientation': "vertical",
        'shrink': 0.8
    }
)

ax.set_xlim(-30, 60) 
ax.set_ylim(30, 72)   
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
scalebar = plt.Line2D([0], [0], color='black', linewidth=2, label='Scalebar (Optional)')
ax.legend([scalebar], ["Approx. Scale"], loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig("postdoc_composite_index_map.png", dpi=300, bbox_inches="tight")
plt.show()

output_file_path = 'university_dataset_with_composite_index.csv'
data_fuzzy_merged_sorted.to_csv(output_file_path, index=False)
