import requests
import pandas as pd
import numpy 
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    with open('train_data.csv', 'wb') as fh:
        fh.write(response.content)
    df = pd.read_csv('train_data.csv', header=None)
    del df[0]
    y = df.iloc[1, :].to_numpy().reshape(-1, 1)
    x = numpy.empty((y.shape[0], 2))
    x[:, 0] = df.iloc[0, :].to_numpy()
    x[:, 1] = 1
    out = numpy.linalg.inv((x.T @ x)) @ x.T @ y
    print(out)
    return area*out[0] + out[1]


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
