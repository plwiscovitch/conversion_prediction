Customer Conversion Prediction

This project explores various machine learning models to determine whether conversion will occur (whether a purchase will be made or not).
Original dataset acquired from https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset.

Model performance was evaluated on F1 score due to the highly imbalanced classes in this dataset. AUC score was used as a secondary metric to determine the most robust model for this classification project. Ultimately, the best model was XGBClassifier with manually tuned parameters providing a f1 score of 0.95, auc score of 0.83, and an accuracy of 0.92. 
