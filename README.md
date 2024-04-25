# Football-Arbitrage-Value-Betting

## Overview
The ML models and betting strategies correspond to Chapter 4 and 5 (Experiment 2 and 3) in the BSc Computer Science thesis titled 'Algorithmic Sports Arbitrage Using Statistical Machine Learning'. The repository contains code for both experiemnts as they were closely related. Experiment 2 consists of a complete machine learning pipeline which utilizes the `data`, `features` and `pipeline` packages. In Experiment 2 the aim was to enhance the predictive accuracy of machine learning models by leveraging novel data transformation methods feature engineering techniques, namely the Pi-rating system, and model evaluation strategies. Experiment 3 focuses on testing the profitability of the machine learning models and betting strategies developed in Experiment 2. The `backtest` notebook contains the code for the strategy definition and backtesting simulation.

## Setup
Clone the repository and run `pip install -r requirements.txt` to install dependencies. Run the following commands: `pip install pipreqs` and `pipreqs --force` to generate requirements.txt file and update dependencies during development.

To run the ML models on the predefined datasets and parameters, run the `main.py` script. To run the backtesting simulation and to define strategies, run and/or the `backtest.ipynb` notebook.


### Data Transformation and Feature Engineering
Key aspects of data preparation include:
- **Pre-processing Techniques**: Ensured data integrity and compatibility with machine learning workflows.
- **Correlation Matrix**: A critical tool for identifying relationships between variables and informing feature selection.
- **Exponential Time-Decay Adjusted Pairwise Ratings**: A novel method that provides a robust and accurate rating of teams, allowing for a nuanced understanding of team dynamics and enhancing match result predictions.
- **SHAP Feature Importance Analysis**: Quantified the predictive power of features, optimizing model performance.

### Model Development and Evaluation
Practical steps involved in model development included:
- **Identifying Relevant Correlations**: Essential for understanding the relationships between different features.
- **Mitigating Overfitting**: Through cross-validation and careful feature selection.
- **Model Assessment**: Using established metrics like accuracy, F1 score, precision, and recall. Notably, the Random Forest Classifier achieved a 56.40% accuracy on the P5 dataset, illustrating its potential for real-world applications.

### Prediction and Betting Strategy
- **Direct Conversion of Model Output to Probabilities**: Allows for measuring confidence and sizing bets accordingly.
- **Dynamic Data Retrieval**: Utilized the Research and Machine Learning Platform to fetch relevant data based on specified feature parameters.

## Inefficiencies in Pre-game Value Betting Markets
- **ROC Curve and AUC**: Employed to identify inefficiencies by comparing the discrepancy between implied probabilities and observed outcomes. AUC values highlighted the predictive power of odds for Home and Away wins versus Draws.

## Strategy Testing and Profitability
- **Backtesting Simulation**: Set up to test the accuracy and profitability of machine learning models (SVM, RFC, XGB) and baseline strategies in the value betting market.
- **Risk Management**: Implemented the Half-Kelly Criterion and confidence thresholds to maximize bankroll while mitigating risk.
- **Longitudinal Testing**: Models were retrained and tested over successive seasons to ensure robust and bias-free evaluation.

### Performance Summary
- **HK-SVM Strategy**: Achieved a profit and loss (PnL) percentage of 52.39% and an accuracy rate of 56.80%, with risk-adjusted return measures like Sharpe ratio of 0.82 and Sortino ratio of 1.14.
- **Comparison with Baseline Strategies**: HK-SVM outperformed others, highlighting the benefits of targeting less significant match outcomes (Away and Draw) for profitable returns.