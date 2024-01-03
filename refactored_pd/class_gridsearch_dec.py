class GridDecision:

    """Alternative method to pruning using grid search hyper parameters optimization"""

    param_grid = {
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Initialize the GridSearchCV object with cross-validation (e.g., k=5)
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
    )

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)
    # print(grid_search.fit(X_train, y_train))

    # Get the best hyperparameters and best estimator
    best_params = grid_search.best_params_
    # print(best_params)
    best_estimator = grid_search.best_estimator_
    print(best_estimator)
