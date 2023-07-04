# Real-time CO2 Emissions Forecasting with Time Series Models

This repository contains the code and resources for extracting real-time CO2 emissions data using an API, cleaning and preprocessing the data, and building various time series models including AR, ARIMA, SARIMA, and LSTM. The repository also includes the necessary scripts for analyzing the models and selecting the best-performing one. Finally, the selected model is used for forecasting CO2 emissions for the next 10 years.

## DataPipeline
<img width="3058" alt="Methodology" src="https://github.com/UdaykiranEstari/real-time-co2-emissions-forecasting/assets/115963773/4118547f-719b-4671-91e7-6d9d9c550425">

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Analysis](#analysis)
- [Forecasting](#forecasting)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/real-time-co2-emissions-forecasting.git
   cd real-time-co2-emissions-forecasting

2. Install the required dependencies:
   ```shell
   pip install -r requirements.txt


## Usage

1. Obtain API credentials:

- Visit the [API Provider]'s website at [API Provider Website URL].
- Sign up for an account and obtain the API credentials.
- Replace the placeholders in the `config.py` file with your API credentials.

2. Extract real-time CO2 emissions data:

- Run the `data_extraction.py` script to extract the latest CO2 emissions data using the API.
- The extracted data will be saved in the `data/` directory.

3. Data preprocessing:

- Use the `data_cleaning.ipynb` Jupyter Notebook to clean and preprocess the data according to your needs.
- The cleaned and preprocessed data will be saved in the `data/processed/` directory.

4. Model building:

- Explore the `models/` directory to find the implementations of AR, ARIMA, SARIMA, and LSTM models.
- Use the `model_selection.ipynb` Jupyter Notebook to compare and evaluate the performance of these models.
- Select the best-performing model based on your analysis.

5. Analysis:

- Utilize the `analysis.ipynb` Jupyter Notebook to further analyze the selected model and gain insights into the CO2 emissions data.
- Visualize the model's predictions, evaluate accuracy metrics, and identify any trends or patterns.

6. Forecasting:

- Once the best model is selected, you can use it to forecast CO2 emissions for the next 10 years.
- Refer to the `forecasting.ipynb` notebook to generate future predictions based on the trained model.

## Contributing

Contributions to this project are welcome. If you find any issues or have ideas for enhancements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
