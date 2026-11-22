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


def run_model(df):
    """Train the tree, compare it with the 20% rule and return everything
    the dashboard and the report need."""
    X, y = build_dataset(df)

    # 70/30 split, stratified so both sets keep the same share of losing lines
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    model = train_tree(X_train, y_train)

    # Model vs baseline (the simple rule "discount above 20%")
    results = pd.DataFrame({
        "Decision tree": scores(y_test, model.predict(X_test)),
        "Rule: discount > 20%": scores(y_test, (X_test["Discount"] > 0.2).astype(int)),
    })

    # ML indicator: predicted loss risk of each line, averaged by sub-category and band
    lines = df.assign(loss_risk=model.predict_proba(X)[:, 1] * 100)
    risk_table = lines.pivot_table(
        index="Sub-Category", columns="Discount Band",
        values="loss_risk", aggfunc="mean",
    ).reindex(columns=ind.BAND_ORDER)

    return {
        "model": model,
        "columns": list(X.columns),
        "test_lines": len(X_test),
        "loss_share": y.mean() * 100,
        "results": results,
        "importance": grouped_importance(model, X.columns),
        "first_split": (X.columns[model.tree_.feature[0]], model.tree_.threshold[0]),
        "risk_table": risk_table,
    }


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    out = run_model(ind.load_clean())

    print(f"Test lines: {out['test_lines']}")
    print(f"Share of losing lines: {out['loss_share']:.1f}%")
    print(f"Accuracy of always saying 'no loss': {100 - out['loss_share']:.1f}%")

    print("\nScores on the loss class (%):")
    print(out["results"].round(1))

    print("\nFeature importance (%):")
    print(out["importance"].round(1))

    feature, threshold = out["first_split"]
    print(f"\nFirst split: {feature} <= {threshold:.3f}")

    print("\nLoss risk by sub-category and discount band (%):")
    print(out["risk_table"].round(0))

    print("\nDecision rules:")
    print(export_text(out["model"], feature_names=out["columns"]))