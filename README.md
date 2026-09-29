# League of Legends Rank Predictor

This project uses Machine Learning to predict a League of Legends player's competitive rank based on tabular gameplay statistics.

The project is part of a Practice Challenge and focuses on building and improving a classification model using gameplay data.

## Project Goal

The main goal is to answer the following question:

> How can Machine Learning be used to predict a League of Legends player's rank using tabular gameplay statistics?

The model uses gameplay statistics such as:

- Kills
- Deaths
- Assists
- CS
- CS per minute
- Gold
- Damage
- Vision
- Objective statistics

The final model will predict a player's rank and the result will later be displayed in a simple web application.

## Machine Learning Process

The project follows these main steps:

1. Load and inspect the data
2. Clean the data
3. Select features and target
4. Preprocess the data
5. Train a baseline model
6. Evaluate the model
7. Improve and compare models
8. Select the final model
9. Predict player rank
10. Display the prediction in a web application

## Project Structure

```text
League Predictor/
├── data/
├── models/
├── notebooks/
├── src/
├── .gitignore
├── README.md
└── requirements.txt