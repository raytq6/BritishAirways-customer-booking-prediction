# British Airways Customer Booking Prediction

Predict whether a customer is likely to complete a booking using a Random Forest model built in Jupyter Notebook.

This project explores the British Airways customer booking dataset, engineers features, trains a baseline model, improves it for class imbalance, and visualizes the results with confusion matrices and feature importance charts.

## Project Overview

The analysis is documented in [Getting Started.ipynb](Getting%20Started.ipynb) and follows this workflow:

1. Load and inspect the dataset.
2. Perform exploratory data analysis.
3. Engineer model-ready features.
4. Train a baseline Random Forest classifier.
5. Improve the model with class balancing and threshold tuning.
6. Review performance with evaluation metrics and visuals.

## Dataset

The dataset used in this project is `customer_booking.csv`. It contains booking-level customer and trip information such as:

- number of passengers
- sales channel
- trip type
- purchase lead time
- length of stay
- flight day and hour
- route and booking origin
- optional extras such as baggage, seat, and meals
- booking completion outcome

## Feature Engineering

The notebook creates a few useful model inputs before training:

- `stay_lead_ratio` to capture the relationship between purchase lead time and length of stay
- one-hot encoding for categorical booking fields such as sales channel and trip type
- cyclical encoding for `flight_day` so the model can interpret the weekly pattern

## Modeling Approach

The main model is a Random Forest classifier. The notebook compares three versions:

- **Baseline model**: a straightforward Random Forest used as the reference point
- **Balanced model**: uses `class_weight='balanced_subsample'` and shallower trees to reduce overfitting and improve minority-class recall
- **Threshold-tuned model**: chooses a better decision threshold from validation data to catch more positive bookings

## Results

The final tuned model improves recall for the completed-booking class, which is the more useful metric for this imbalanced problem.

Saved summary metrics in `rf_summary.txt` show the broader cross-validation view of the model, and the notebook records the more detailed train/test comparison.

## Visual Outputs

The repo includes saved charts used in the analysis:

- `rf_confusion.png`
- `rf_importances.png`
- `rf_roc.png`

These visuals help explain where the model gets predictions right or wrong and which features matter most.

## Repository Structure

- `Getting Started.ipynb` - main analysis notebook
- `customer_booking.csv` - input dataset
- `rf_summary.txt` - model summary metrics
- `rf_confusion.png` - confusion matrix visual
- `rf_importances.png` - feature importance visual
- `rf_roc.png` - ROC curve visual
- `requirements.txt` - Python dependencies

## Setup and Run

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Open `Getting Started.ipynb` in Jupyter or VS Code and run the cells from top to bottom.

## Dependencies

The notebook uses:

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
