# Machine Learning: 
Classification & PreprocessingThis repository contains implementations and experiments covering the end-to-end machine learning pipeline, from data preparation to model evaluation.

# Core Topics
1. Data PreprocessingFeature Scaling:<br>
    1. Implementation of StandardScaler to normalize feature distributions ($mean = 0, std = 1$). Essential for distance-based and gradient-based algorithms.
    2. Handling Class Imbalance:<br>
        Utilizing RandomOverSampler (from imblearn) to balance training sets, ensuring the models do not become biased toward the majority class.
    3. Data Splitting:<br>
        Custom workflows for partitioning data into Training, Validation, and Testing sets to ensure model generalizability.<br><br>

2. Models ImplementedThe following classification algorithms are implemented and compared:
    1. K-Nearest Neighbors (KNN):<br>
        Distance-based classification using $k$-neighbor voting.
    2. Naive Bayes: <br>
        Probabilistic classification based on Bayes' Theorem with feature independence assumptions.
    3. Logistic Regression: <br>
        Linear model for predicting categorical outcomes via the sigmoid function.
    4. Support Vector Machines (SVM):<br>
        Maximum-margin classification using optimal hyperplanes.
    5. Neural Networks (NN):<br>
        Multi-layer perceptron implementations for capturing non-linear relationships.<br><br>

3. Evaluation MetricsModels are evaluated using metrics beyond simple accuracy to account for imbalanced datasets:
    Precision & RecallF1 Score (Harmonic mean of precision and recall)Confusion Matrix Analysis

# Tools UsedLanguage:
 1. PythonData: Pandas, NumPy
 2. Visualization: Matplotlib, Seaborn
 3. Machine Learning: Scikit-learn, Imbalanced-learn, TensorFlow/Keras (for NN)