import pandas as pd

input_path = 'data/weather_forecast_data.csv'

data = pd.read_csv(input_path)

data['Rain'] = data['Rain'].replace({'rain': True, 'no rain': False})

output_path = 'data/weather_forecast_data_updated.csv'

data.to_csv(output_path, index=False)

print(f"Arquivo atualizado salvo em: {output_path}")
