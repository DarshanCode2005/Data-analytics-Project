# India CO₂ Emissions Forecasting & Data Analytics

## 🌍 Project Overview

This project focuses on forecasting **India's CO₂ emissions from 2016 to 2070** using advanced **time series modeling** techniques. The analysis provides insights into different policy scenarios and their impact on emissions reduction, helping in understanding India's pathway to achieving **net-zero emissions**.

## 📌 Key Objectives

- **Analyze past CO₂ emissions trends** using historical data (2000-2015).
- **Develop predictive models** (LSTM, CNN-GRU-LSTM Hybrid, Transformer, and Kaya Identity model) to forecast emissions till 2070.
- **Evaluate different policy scenarios**: Business-as-Usual, Balanced Renewable Push, and Aggressive Renewable Growth.
- **Perform detailed data analytics** to support the findings.

## 🛠️ Solution Approach

### 🔹 1. **Data Preprocessing & Visualization**

- Loaded and cleaned **India’s CO₂ emissions dataset** (2000-2015).
- Scaled data using **MinMaxScaler** for machine learning models.
- Created interactive **visualizations** to explore emissions trends.

### 🔹 2. **LSTM-Based Forecasting**

- Implemented **Long Short-Term Memory (LSTM)** model for time series forecasting.
- Used **early stopping & model checkpointing** for optimization.
- Plotted **training vs validation loss** graphs.
- Found that LSTM provides a good trend analysis but leads to stagnation after 2040.

### 🔹 3. **Hybrid CNN-GRU-LSTM Model**

- Combined **Convolutional Neural Networks (CNN)** for feature extraction.
- Used **Gated Recurrent Units (GRU)** for efficient sequential learning.
- Integrated **LSTM layers** for long-term dependencies.
- Achieved better accuracy, but slight fluctuations in forecasts were observed.

### 🔹 4. **Kaya Identity Model** (Deterministic Forecasting)

- Applied **Kaya Identity** to estimate CO₂ emissions using:
  - **Population Growth**
  - **GDP per Capita Growth**
  - **Energy Intensity**
  - **Carbon Intensity**
- Forecasted emissions for three scenarios:
  - **Business-as-Usual** (Steady increase, peaking at \~5 metric tons/capita)
  - **Balanced Renewable Push** (Peaks & declines by 2070)
  - **Aggressive Renewable Growth** (Declines to net zero by 2040)

## 📊 Data Analytics & Monte Carlo Simulation

- Performed **descriptive statistics** on CO₂ emissions data.
- Conducted **correlation analysis** between emissions and economic indicators.
- Applied **Monte Carlo Simulation** to estimate uncertainty in forecasts.
- Used **distribution fitting** to analyze emission trends.

## 🔍 Key Findings

- **Business-as-Usual** scenario shows CO₂ emissions continuing to rise.
- **Aggressive Renewable Push** scenario can achieve net zero by 2040.
- **LSTM & Hybrid models** provide better predictions than traditional methods.
- **Transformer model** offers better stability but requires fine-tuning.
- **Policy interventions** like renewable energy adoption & efficiency improvements are crucial.

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/DarshanCode2005/Data-analytics-Project.git
   cd Data-analytics-Project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open **Submission.ipynb** & execute the cells.

## 📚 Research References

- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation.*
- Vaswani, A. et al. (2017). "Attention is All You Need." *NeurIPS.*
- Kaya, Y. (1990). "Impact of Carbon Emissions on Economic Growth." *IPCC Report.*

---

📩 **For any queries, feel free to reach out!**

## 📌 Colab Notebook

The entire Colab notebook is available at:
[Open in Google Colab](https://colab.research.google.com/drive/1odKAx7h5r5GZHx_XFGiJQf-FSKfpW45W?usp=sharing)

## 📦 Requirements

To install all dependencies, run the following command:

```bash
pip install -r requirements.txt
```

### Dependencies:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
keras
torch
statsmodels
xgboost
scipy
plotly
ipywidgets
transformers
nltk
rich
keras-tuner
jupyter
jupyterlab
notebook
geopandas
folium
google-generativeai
```

This ensures all required packages are installed for running the project seamlessly.🚀

