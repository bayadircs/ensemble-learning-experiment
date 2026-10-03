import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, StackingClassifier

SEED = 42

def disagreement(predictions):
    keys = list(predictions.keys())
    return [
        (keys[i], keys[j], (predictions[keys[i]] != predictions[keys[j]]).mean())
        for i in range(len(keys))
        for j in range(i + 1, len(keys))
    ]

def make_data(seed=SEED, n_rows=10000, n_cols=70):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_cols))

    trend = X[:, 0] + X[:, 1] ** 2 - X[:, 2] + 0.8 * X[:, 3]
    rule = (X[:, 0] > 0) & (X[:, 1] > 0) | (X[:, 0] < -0.5) & (X[:, 1] < 0) & (X[:, 2] > 0) & (X[:, 3] < 0)

    yes_chance = 1 / (1 + np.exp(-(trend + rule)))
    y = (rng.random(n_rows) < yes_chance).astype(int)

    ceiling = np.maximum(yes_chance, 1 - yes_chance).mean()
    return X, y, ceiling

def main():
    X, y, CEILING = make_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.24, random_state=SEED
    )

    # المجموعة الأولى: نماذج متنوعة (Linear + Tree + KNN)
    model_crew = {
        'line': make_pipeline(StandardScaler(), LogisticRegression()),
        'tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, random_state=SEED),
        'near': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=50))
    }

    guesses = {
        name: model.fit(X_train, y_train).predict(X_test)
        for name, model in model_crew.items()
    }

    best_alone = max(guesses.values(), key=lambda p: (p == y_test).mean())
    print(f"best single model: {round((best_alone == y_test).mean() * 100, 1)}%")

    print("\nhow often each pair gives a diff answer(%):")
    for a, b, p in disagreement(guesses):
        print(f"{a} vs {b}: {round(p * 100, 1)}%")

    crew_gap = np.mean([p for _, _, p in disagreement(guesses)])

    # Voting Classifier
    voting = VotingClassifier(list(model_crew.items()), voting='soft')
    voting.fit(X_train, y_train)

    # Stacking Classifier
    stacking = StackingClassifier(
        list(model_crew.items()),
        final_estimator=LogisticRegression(),
        cv=5
    )
    stacking.fit(X_train, y_train)

    print(f"\nvoting (round-voting score): {round(voting.score(X_test, y_test) * 100, 1)}%")
    print(f"stacking (round-stacking score): {round(stacking.score(X_test, y_test) * 100, 1)}%")
    print(f"best single was {round((best_alone == y_test).mean() * 100, 1)}% || ceiling is round(CEILING): {round(CEILING * 100, 1)}%")

    # المجموعة الثانية: نماذج خطية فقط (Linear-based models)
    lookalikes = {
        'logreg': make_pipeline(StandardScaler(), LogisticRegression()),
        'sgd': make_pipeline(StandardScaler(), SGDClassifier(loss='log_loss', random_state=SEED)),
        'bayes': make_pipeline(StandardScaler(), GaussianNB())
    }

    look_guesses = {
        name: model.fit(X_train, y_train).predict(X_test)
        for name, model in lookalikes.items()
    }

    look_gap = np.mean([p for _, _, p in disagreement(look_guesses)])
    print(f"\navg disagreement: {round(look_gap * 100, 1)}% || {round(crew_gap * 100, 1)}% (the crew above: {round(crew_gap * 100, 1)}%)")

    look_vote = VotingClassifier(list(lookalikes.items()), voting='soft')
    look_vote.fit(X_train, y_train)
    look_best = max([ (p == y_test).mean() for p in look_guesses.values() ])

    print(f"best single: {round(look_best * 100, 1)}%")
    print(f"voting: {round(look_vote.score(X_test, y_test) * 100, 1)}%")

if __name__ == '__main__':
    main()