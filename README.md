#AI- BASED-DIET-PLAN-RECOMMENDATION-
This project implements an AI-based diet recommendation system designed to generate personalized diet plans based on user inputs like weight, age, activity level, medical conditions, and dietary preferences.
The system uses a machine learning algorithm to classify users into different categories and recommends suitable diet types.

Table of Contents

Project Description
Technologies Used
Features
How It Works
Setup & Installation
Contact


Project Description
This project uses data from various users, including their weight, height, age, gender, and medical history, to predict the best diet plan for each individual. Using a Random Forest Classifier, the system classifies users based on features such as their BMI, BMR, activity level, and dietary restrictions. The model provides users with an optimal calorie intake and recommends a diet type (e.g., Vegan, Non-Vegetarian) suited to their goals (e.g., weight loss, muscle gain).

Goal:
To recommend a personalized diet plan.
To improve the overall health of individuals by providing tailored nutritional advice.


Technologies Used
Python: Primary programming language.
pandas: For data cleaning and manipulation.
scikit-learn: For model training, testing, and evaluation.
matplotlib: For data visualization (e.g., feature importance chart).
joblib: for saving the trained model
Jupyter Notebooks: For model development and testing.
RandomForestClassifier: Machine learning model used for predictions.


Features
Personalized Diet Plans: Based on user's personal data (age, weight, activity level, etc.).
Health Considerations: Takes into account medical history like cholesterol level and blood pressure.
Activity Level: Considers the user’s activity level for more accurate calorie intake predictions.
Diet Type Recommendations: Recommends diet plans (e.g., Vegan, Non-Vegetarian, Vegetarian) based on goals (weight loss, muscle gain).


How It Works
Data Collection: The system collects user data like age, weight, height, gender, and more.
Preprocessing: The input data is cleaned, transformed, and normalized (if needed).
Feature Engineering: Important features are selected and one-hot encoding is applied for categorical variables.
Model Training: A Random Forest Classifier is trained using labeled data to predict diet categories.
User Input: Users can enter their details, and the model predicts the appropriate diet plan and calorie intake.
Output: The system displays the personalized diet plan, including daily calorie intake and recommended food types.


Setup & Installation
To use this project locally, follow these steps:
Clone the repository:


Feel free to reach out with any questions or suggestions:

GitHub: https://github.com/nehin17
Email: junewalneha@gmail.com













