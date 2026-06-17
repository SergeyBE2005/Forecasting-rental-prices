# Forecasting Rental Prices

**Forecasting-rental-prices**

This project demonstrates a machine learning pipeline for forecasting housing/rental prices. It uses the California Housing dataset to train a Random Forest Regressor and evaluate its performance. The model is saved for future predictions.

## Project Goal

The primary objective is to build and train a regression model to predict the median house value for a given block in California based on various features like median income, house age, and average rooms.

## Project Structure

This repository contains the following key files:

- **Forecasting-rental-prices.ipynb**: The main Jupyter Notebook containing the entire workflow. It includes:
  - Data loading using sklearn.datasets.fetch_california_housing
  - Exploratory Data Analysis (EDA) with correlation heatmaps and scatter plots
  - Data preprocessing and feature selection
  - Model training using a Random Forest Regressor
  - Model evaluation using the R2 Score
  - Code to save the trained model
- **model.pkl**: The saved pre-trained Random Forest model (created by the notebook)
- **requirements.txt**: A list of all Python dependencies required to run the notebook
- **dockerfile**: A Dockerfile to build a containerized environment for the project
- **correlation.png**: A saved correlation matrix visualization
- **scatter.png**: A saved scatter plot showing the relationship between median income and price
- **README.md**: This file (in English)
- **README.ru.md**: A Russian version of this README

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- (Optional) Docker to run the project in a container

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/SergeyBE2005/Forecasting-rental-prices.git
   cd Forecasting-rental-prices
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Notebook

To explore the code and train the model:

```
jupyter notebook Forecasting-rental-prices.ipynb
```

Run the cells in the notebook sequentially to:
1. Load and explore the California Housing data
2. Visualize feature correlations and relationships
3. Train the Random Forest model
4. Evaluate the model's accuracy
5. Save the trained model to model.pkl

### Running with Docker

To build and run the project in a containerized environment:

1. Build the Docker image:
   ```
   docker build -t forecasting-rental-prices .
   ```

2. Run the container:
   ```
   docker run -p 8888:8888 forecasting-rental-prices
   ```
   This will start a Jupyter Notebook server inside the container. Access it in your browser at http://localhost:8888

## Model Performance

The current model, a Random Forest Regressor with 100 estimators, achieves an R2 Score of approximately 0.8141 on the test set. This indicates that the model explains about 81.4% of the variance in the target variable (house prices).

## Usage

To use the saved model for making predictions on new data:

1. Ensure you have the model.pkl file
2. Use the following Python script as an example:

```python
import joblib
import numpy as np

# Load the model
model = joblib.load('model.pkl')

# Example: Predict price for a new sample (8 features from the dataset)
# [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
new_data = np.array([[8.0, 45.0, 6.0, 1.0, 1000.0, 3.0, 37.0, -122.0]])
predicted_price = model.predict(new_data)

print(f'Predicted Price: {predicted_price[0]:.2f}')
```

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request or open an Issue to improve the project.
