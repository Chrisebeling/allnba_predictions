To run the data prep notebook, the package nba_stats is required. This is a custom package which connect to a database of nba stats. This can be done easily using the below line but access to the database itself will need to be arranged for any of the connections to work. This is easy enough and happy to give access, just let me know.

pip install git+https://github.com/Chrisebeling/nba_stats.git

## Summary
1. Data Prep - imports from database, cleansing, validation of data, feature selection
2. Train XGBoost - run grid search on training data to select hyper params based on val results, a number of visualisations using shap etc to assess model based on test data.
3. Results - Final model is then run on 2021 season to get predicition for allnba team. This was done before the team was selected, so the actual team is not included as comparison.

**Predict function** - Instead of using the default grid search and predict from sklearn, custom functions were implemented in order to account for the difficulty of restricting predictions to combinations of 6G,6F,3C per season. In addition, logic is implemented to account for the position that a player is selected in when players have multiple eligible positions. By implementing in this way, the logic can be incorporated at the training stage, ensuring that the model is trained on the same basis as what we are trying to predict. This does improve the model's accuracy (but not dramatically) and removes some instances where impossible predictions are made (i.e. 8 forwards in a season).

**Current params:** {'learning_rate':0.07, 'max_depth': 3, 'n_estimators': 400, 'reg_lambda': 7.5, 'subsample': 0.6}

**Current features:** 'ast', 'blk', 'drb', 'fg', 'fg3', 'ft', 'mp', 'orb', 'pf', 'plus_minus', 'pts', 'stl', 'tov', 'fg_pct', 'fg3_pct', 'ft_pct', 'fg2', 'fg2_pct', 'assist_tov', 'ts_pct', 'ast_pct', 'blk_pct', 'def_rtg', 'drb_pct', 'off_rtg', 'orb_pct', 'stl_pct', 'tov_pct', 'trb_pct', 'usg_pct', 'W_pct', 'game_pct'
