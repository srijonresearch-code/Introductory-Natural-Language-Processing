# Introductory Natural Language Processing: SMS Spam Classifier

My work from Kaggle's **Introductory Natural Language Processing** course: a spam classifier that labels SMS messages as **spam** or **ham** (normal), built with NLTK and XGBoost.

## Dataset

SMS Spam Collection (`spam.csv`): text messages labeled `spam` or `ham`. The file needs `encoding='latin-1'` to load because it contains non-standard characters. The classes are imbalanced, with far more ham than spam.

## Approach

1. **Lowercasing** so "Free" and "free" count as the same word
2. **Punctuation removal** with a regex (`[^\w\s]`), so "winner!" and "winner" match
3. **Tokenization** with NLTK's `word_tokenize`, splitting each message into a list of words
4. **Stop word removal** using NLTK's English stop word list, stored as a `set` for fast lookups
5. **Vectorization** to turn the cleaned tokens into numeric features the model can use
6. **Modeling** with `XGBClassifier`, using `scale_pos_weight` to account for the class imbalance, `random_state` for reproducibility, and `logloss` as the evaluation metric
7. **Evaluation** with accuracy, precision, recall, F1, and a confusion matrix heatmap
8. **Saving artifacts**: the trained model with `joblib`, and the test labels and predictions with `numpy` and `scipy`

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.973 |
| Precision | 0.940 |
| Recall | 0.862 |
| F1 | 0.899 |

## Key Takeaway

Accuracy alone is misleading on imbalanced data. Most messages are ham, so a model that never predicted spam would already score high accuracy while catching nothing. Precision and recall show what actually matters:

- **Precision (94%):** when the model flags a message as spam, it is right 94% of the time
- **Recall (86%):** the model catches 86% of the real spam, so some still gets through

Raising `scale_pos_weight` trades precision for recall, so the right value depends on which mistake is worse: blocking a real message or letting spam through.

## Tools & Libraries

`pandas` · `NLTK` · `scikit-learn` · `XGBoost` · `seaborn` · `matplotlib` · `joblib` . `Scipy`


---
*Part of my self-directed path toward becoming a Machine Learning Engineer.*
