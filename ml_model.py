import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

import indicators as ind

# Inputs known at the time of sale. Never Profit: it is the answer.
FEATURES_NUM = ["Discount", "Quantity", "Order Month"]
FEATURES_CAT = ["Sub-Category", "Region", "Segment", "Ship Mode"]
RANDOM_STATE = 42


def build_dataset(df):
    """X: model inputs (categories turned into 0/1 columns), y: 1 if the line loses money."""
    X = pd.get_dummies(df[FEATURES_NUM + FEATURES_CAT], columns=FEATURES_CAT)
    y = (df["Profit"] < 0).astype(int)
    return X, y


def train_tree(X_train, y_train, max_depth=4):
    """Decision tree with a limited depth, so its rules stay short and readable."""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def scores(y_true, y_pred):
    """Precision, recall and F1 on the loss class, in %."""
    return {
        "precision": precision_score(y_true, y_pred) * 100,
        "recall": recall_score(y_true, y_pred) * 100,
        "f1": f1_score(y_true, y_pred) * 100,
    }


def grouped_importance(model, columns):
    """Feature importance summed back to the original columns, in %."""
    importance = pd.Series(model.feature_importances_, index=columns)
    all_features = FEATURES_NUM + FEATURES_CAT
    original = [
        next(f for f in all_features if col == f or col.startswith(f + "_"))
        for col in columns
    ]
    return importance.groupby(original).sum().sort_values(ascending=False) * 100


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df = ind.load_clean()
    X, y = build_dataset(df)

    # 70/30 split, stratified so both sets keep the same share of losing lines
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Test lines: {len(X_test)}")
    print(f"Share of losing lines: {y.mean() * 100:.1f}%")
    print(f"Accuracy of always saying 'no loss': {(1 - y.mean()) * 100:.1f}%")

    # ----- Model -----
    model = train_tree(X_train, y_train)
    tree_scores = scores(y_test, model.predict(X_test))

    # ----- Baseline: the simple rule "discount above 20%" -----
    rule_pred = (X_test["Discount"] > 0.2).astype(int)
    rule_scores = scores(y_test, rule_pred)

    print("\nScores on the loss class (%):")
    results = pd.DataFrame({"Decision tree": tree_scores,
                            "Rule: discount > 20%": rule_scores})
    print(results.round(1))

    # ----- What drives the model -----
    print("\nFeature importance (%):")
    print(grouped_importance(model, X.columns).round(1))

    first_feature = X.columns[model.tree_.feature[0]]
    first_threshold = model.tree_.threshold[0]
    print(f"\nFirst split: {first_feature} <= {first_threshold:.3f}")

    print("\nDecision rules:")
    print(export_text(model, feature_names=list(X.columns)))