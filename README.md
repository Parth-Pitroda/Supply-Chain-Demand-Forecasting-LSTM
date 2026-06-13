"# Supply-Chain-Demand-Forecasting-LSTM" 
# Supply Chain Demand Forecasting with LSTM

## 📌 Project Overview
In modern supply chain networks, under-predicting demand leads to costly stockouts, while over-predicting leads to wasted warehouse space and capital. This project implements a Deep Learning sequence-to-sequence model to forecast retail demand over a 7-day horizon, based on 30 days of historical sales data.

This project was built to demonstrate proficiency in **Sequential Learning, Time-Series Feature Engineering, and PyTorch Neural Network Design.**

## 📊 The Dataset
The model is trained on the industry-standard **[Kaggle Store Item Demand Forecasting Dataset](https://www.kaggle.com/competitions/demand-forecasting-kernels-only)**. 
* Contains 5 years of real-world daily sales data.
* Encompasses 10 different retail stores and 50 unique items.
* Total dataset size: 913,000 daily transaction records.

## 🏗️ Model Architecture
Instead of treating time-series as a standard tabular regression problem, this project utilizes an **Encoder-Decoder LSTM (Long Short-Term Memory)** network built from scratch in PyTorch.

* **The Sliding Window:** Chronological data is dynamically sliced into a Lookback Window ($W = 30$ days) to predict a continuous Forecast Horizon ($H = 7$ days).
* **The Encoder:** Processes the historical sequence and captures underlying trends and dependencies.
* **The Decoder:** Autoregressively steps through the 7-day horizon, predicting one day at a time while updating its internal state.

## ⚙️ Feature Engineering
Raw sales counts were augmented with temporal metadata to help the neural network understand human shopping cycles:
* **Temporal Cyclicality:** Extracted `day_of_week` to map weekly rhythms.
* **Binary Indicators:** Created an `is_weekend` flag, as retail volume heavily skews toward Saturdays and Sundays.

## 📈 Evaluation & Business Metric
Standard Mean Squared Error (MSE) is used for backpropagation, but the model's true performance is evaluated using **WMAPE (Weighted Mean Absolute Percentage Error)**. 

WMAPE is an industry-standard metric in retail because it inherently weighs high-volume items more heavily than low-volume items, giving a true representation of business impact.

$$\text{WMAPE} = \frac{\sum_{i=1}^{N} |y_i - \hat{y}_i|}{\sum_{i=1}^{N} |y_i|}$$

### **Final Results**
After training the model on the localized data of Store 1 / Item 1 for 15 epochs, the model successfully mapped the human shopping patterns, driving the validation error down to **~21.18% WMAPE**, establishing a highly robust baseline for single-item forecasting.

## 🚀 How to Run the Code
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/Parth-Pitroda/Supply-Chain-Demand-Forecasting-LSTM.git](https://github.com/Parth-Pitroda/Supply-Chain-Demand-Forecasting-LSTM.git)
