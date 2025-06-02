import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

pokemon_stats = pd.read_csv("pokemon_summary.csv")
pokemon_data = pd.read_csv("main_pkmn_dataset.csv")
trainers = pd.read_csv("trainers.csv")
exp_types = pd.read_csv("exp_types.csv")

trainer_pokemon = trainers[['trainer.name', 'pokemon1', 'pokemon2', 'pokemon3', 'pokemon4', 'pokemon5', 'pokemon6',
                           'level_pokemon1', 'level_pokemon2', 'level_pokemon3', 'level_pokemon4', 'level_pokemon5', 'level_pokemon6', 'total_EXP']].copy()

# Merge stats for each Pokémon
for i in range(1, 7):
    pokemon_col = f'pokemon{i}'
    level_col = f'level_pokemon{i}'
    temp = trainer_pokemon[[pokemon_col, level_col, 'trainer.name']].merge(
        pokemon_stats[['name', 'attack', 'defense', 'hp', 'sp_attack', 'sp_defense', 'speed']],
        left_on=pokemon_col, right_on='name', how='left'
    )
    trainer_pokemon[f'attack{i}'] = temp['attack']
    trainer_pokemon[f'defense{i}'] = temp['defense']
    trainer_pokemon[f'hp{i}'] = temp['hp']
    trainer_pokemon[f'sp_attack{i}'] = temp['sp_attack']
    trainer_pokemon[f'sp_defense{i}'] = temp['sp_defense']
    trainer_pokemon[f'speed{i}'] = temp['speed']

trainer_pokemon = trainer_pokemon.reset_index(drop=True)

features = [f'{stat}{i}' for i in range(1, 7) for stat in ['attack', 'defense', 'hp', 'sp_attack', 'sp_defense', 'speed']] + [f'level_pokemon{i}' for i in range(1, 7)]
X = trainer_pokemon[features].fillna(0)
y = trainer_pokemon['total_EXP']

# Normalize 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Polynomial regression
degree = 2
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(X_scaled, y)

# Predictions 
y_pred = polyreg.predict(X_scaled)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"MSE: {mse:.2f}, R-squared: {r2:.2f}")

# Calculate difficulty and predicted EXP
trainer_pokemon['predicted_EXP'] = y_pred
trainer_pokemon['difficulty'] = trainer_pokemon[[f'{stat}{i}' for i in range(1, 7) for stat in ['attack', 'defense', 'hp', 'sp_attack', 'sp_defense', 'speed']]].sum(axis=1) + trainer_pokemon[[f'level_pokemon{i}' for i in range(1, 7)]].sum(axis=1)
trainer_pokemon['cost_effectiveness'] = trainer_pokemon['predicted_EXP'] / trainer_pokemon['difficulty']

# Plot 1: Predicted EXP vs. Actual EXP
plt.figure(figsize=(8, 6))
plt.scatter(trainer_pokemon['total_EXP'], trainer_pokemon['predicted_EXP'], color='blue', alpha=0.5)
plt.plot([trainer_pokemon['total_EXP'].min(), trainer_pokemon['total_EXP'].max()],
         [trainer_pokemon['total_EXP'].min(), trainer_pokemon['total_EXP'].max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual EXP (total_EXP)')
plt.ylabel('Predicted EXP')
plt.title(f'Predicted EXP vs. Actual EXP\nMSE: {mse:.2f}, R-squared: {r2:.2f}')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Predicted EXP vs. Difficulty
plt.figure(figsize=(8, 6))
plt.scatter(trainer_pokemon['difficulty'], trainer_pokemon['predicted_EXP'], color='green', alpha=0.5)
plt.xlabel('Difficulty (Sum of Stats + Levels)')
plt.ylabel('Predicted EXP')
plt.title('Predicted EXP vs. Difficulty')
plt.grid(True)
plt.show()

# Plot 3: Actual EXP vs. Difficulty
plt.figure(figsize=(8, 6))
plt.scatter(trainer_pokemon['difficulty'], trainer_pokemon['total_EXP'], color='purple', alpha=0.5)
plt.xlabel('Difficulty (Sum of Stats + Levels)')
plt.ylabel('Actual EXP (total_EXP)')
plt.title('Actual EXP vs. Difficulty')
plt.grid(True)
plt.show()

# Example
pokemon_name = 'Grotle'
target_level = 38

# Current level
current_level = pokemon_data[pokemon_data['name'] == pokemon_name]['Level'].iloc[0]
growth_group = pokemon_stats[pokemon_stats['name'] == pokemon_name]['experience_growth'].iloc[0]

growth_group_map = {
    600000: 'Erratic',
    800000: 'Fast',
    1000000: 'Medium Fast',
    1059860: 'Medium Slow',
    1250000: 'Slow',
    1640000: 'Fluctuating'
}
growth_column = growth_group_map.get(growth_group, 'Medium Slow')

growth_to_next_level_map = {
    'Erratic': 'To next level',
    'Fast': 'To next level.1',
    'Medium Fast': 'To next level.2',
    'Medium Slow': 'To next level.3',
    'Slow': 'To next level.4',
    'Fluctuating': 'To next level.5'
}
exp_column = growth_to_next_level_map.get(growth_column, 'To next level.3')

# Get EXP needed 
exp_needed = 0
if target_level > current_level:
    for level in range(int(current_level), target_level):
        exp_row = exp_types[exp_types['Unnamed: 6_level_0'] == str(level)]
        if not exp_row.empty:
            exp_needed += float(exp_row[exp_column].iloc[0])
        else:
            raise ValueError(f"No EXP data found for level {level} in exp_types.csv.")
else:
    print(f"Target level {target_level} is not higher than current level {current_level}. No EXP needed.")

# Pokémon levels <= target_level + 5
max_level = target_level + 5
level_columns = [f'level_pokemon{i}' for i in range(1, 7)]
valid_trainers = trainer_pokemon[level_columns].apply(lambda x: x[x.notna()].max() <= max_level, axis=1)
trainer_pokemon = trainer_pokemon.loc[valid_trainers]

# Select trainers
trainer_pokemon = trainer_pokemon.sort_values('cost_effectiveness', ascending=False)
selected_trainers = trainer_pokemon[trainer_pokemon['predicted_EXP'].cumsum() >= exp_needed] if exp_needed > 0 else trainer_pokemon.head()

# Output
output_columns = ['trainer.name', 'pokemon1', 'level_pokemon1', 'pokemon2', 'level_pokemon2', 
                  'pokemon3', 'level_pokemon3', 'pokemon4', 'level_pokemon4', 
                  'pokemon5', 'level_pokemon5', 'pokemon6', 'level_pokemon6', 'predicted_EXP']
print(f"Optimal trainers to battle for {pokemon_name} to reach level {target_level}:")
print(selected_trainers[output_columns].head())