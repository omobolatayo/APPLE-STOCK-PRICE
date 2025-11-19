This project focuses on predicting the future closing price of Apple Inc. (AAPL) stock using historical data and a machine-learning model. The data was collected from Yahoo Finance and includes daily closing prices from 2020 to the present.

To model the stock movement, the closing prices were normalized and converted into time-series sequences, where each sequence contains 29 past days used to predict the price of the next day. A Long Short-Term Memory (LSTM) neural network was trained because LSTMs are effective for sequential data and can capture trends and patterns over time.

The model was built using PyTorch and trained with the Mean Squared Error (MSE) loss function. After training, predictions were compared with actual prices using common accuracy measures such as MAE and RMSE. The predicted and actual values were also visualized to show how closely the model follows real market behavior.

Overall, the project demonstrates how deep learning can be applied to financial time-series forecasting, providing insights into stock trends, though not guaranteeing future performance.
