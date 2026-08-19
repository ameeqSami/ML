# Breast Cancer Diagnosis

This project builds a classification model to predict whether a breast tumor is malignant or benign, based on measurements taken from digitized biopsy images. It uses the Wisconsin Breast Cancer dataset and achieves 99.12% accuracy on the test set.

---

## Background

The dataset was originally compiled by Dr. William H. Wolberg at the University of Wisconsin and is one of the most well-known datasets in medical machine learning. Each row represents a single biopsy, and the features capture geometric and texture properties of the cell nuclei found in the image — things like radius, smoothness, and concavity.

The goal is straightforward: given those measurements, predict whether the tumor is malignant (M) or benign (B).

---

## Dataset

- **Source:** UCI Machine Learning Repository (Wisconsin Breast Cancer Diagnostic)
- **Samples:** 569
- **Features:** 30 numerical features (after dropping the id and an empty trailing column)
- **Target:** diagnosis — M (Malignant) or B (Benign)

The 30 features are derived from 10 base measurements, each recorded as a mean, standard error, and worst-case value across the nuclei in the image:

- Radius, Texture, Perimeter, Area
- Smoothness, Compactness, Concavity
- Concave Points, Symmetry, Fractal Dimension

---

## Approach

Rather than throwing all 30 features directly into a model, the pipeline first filters out features that have little correlation with the target, then scales the remaining ones and reduces dimensionality with PCA before fitting a logistic regression classifier.

The full pipeline looks like this:

    1. CorrelationThresholdFilter  -- removes features with |correlation| < 0.10
    2. StandardScaler              -- normalizes features to zero mean and unit variance
    3. PCA (95% variance)          -- reduces dimensions while retaining most information
    4. LogisticRegression          -- the final classifier

The correlation filter (step 1) is a custom sklearn transformer written in ctf.py. It computes the Pearson correlation of each feature against the target during fit, then drops anything that falls below the threshold. This keeps only the features that actually have a meaningful linear relationship with the outcome, which simplifies the downstream steps without needing manual feature selection.

---

## Results

The pipeline was trained on 80% of the data and evaluated on the remaining 20%.

    Test Accuracy: 99.12%

---

## Files

    breast_cancer.ipynb              -- main notebook with EDA, preprocessing, and training
    ctf.py                           -- CorrelationThresholdFilter custom transformer
    cancer_diagnosis_pipeline.joblib -- the trained and serialized pipeline
    data.csv                         -- the raw dataset

---

## Running It

Install the dependencies:

    pip install pandas numpy scikit-learn joblib

Then open the notebook:

    jupyter notebook breast_cancer.ipynb

To load the saved model and run predictions on new data:

    import joblib

    pipe = joblib.load("cancer_diagnosis_pipeline.joblib")
    predictions = pipe.predict(X_new)

Input to the model should be a DataFrame or array with the same 30 features as the original dataset, in the same column order (everything except id and the empty Unnamed: 32 column).

---

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- joblib
