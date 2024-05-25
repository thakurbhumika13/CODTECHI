# CODTECHI
1. Importing Necessary Libraries
The implementation requires the following libraries:

NumPy: For numerical computing.
pandas: For data manipulation and analysis.
scikit-learn: For machine learning algorithms and tools.
2. Loading the Dataset
The Boston Housing dataset is loaded using scikit-learn's load_boston() function. The dataset contains various features related to housing characteristics and the target variable, which is the median value of owner-occupied homes.

3. Selecting Features
The features selected for predicting housing prices include:

RM (average number of rooms per dwelling)
LSTAT (percentage of lower status of the population)
CRIM (per capita crime rate by town)
These features are chosen based on their potential impact on housing prices.

4. Splitting the Dataset
The dataset is split into training and testing sets using scikit-learn's train_test_split() function. This step ensures that the model's performance can be evaluated on unseen data.

5. Training the Linear Regression Model
A linear regression model is trained using scikit-learn's LinearRegression() class. The model is trained on the selected features and the training data.

6. Making Predictions
Once the model is trained, it is used to make predictions on the testing data using the predict() method of the trained model.

7. Evaluating the Model
The performance of the trained model is evaluated using the following metrics:

Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R-squared score
These metrics provide insights into how well the model performs in predicting housing prices.

Usage
To use the implementation:

Ensure that the required libraries (numpy, pandas, scikit-learn) are installed.
Copy and paste the provided code snippet into your Python environment.
Run the script to train the linear regression model and evaluate its performance.
