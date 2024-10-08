import pandas as pd
from pycaret.classification import load_model, predict_model


def predict_churn(data: pd.DataFrame) -> pd.DataFrame:

    # Load saved churn model
    model = load_model('churn_model')

    # Use the model to predict probabilities on the data provided
    predictions = predict_model(model, data=data)

    return predictions[['prediction_label', 'prediction_core']]

if __name__ == "__main__":

    # Get the data from csv
    new_data = pd.read_csv('new_churn_data.csv')

    # Get our predictions
    predictions = predict_churn(new_data)

    # Print
    print(predictions)