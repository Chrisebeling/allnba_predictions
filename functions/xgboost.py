import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import sklearn as sk
import progressbar

def season_ttv_split(X, Y, id_data, test_size, split_val=False):
    m, n = X.shape
    season_series = id_data['season']
    seasons = list(set(season_series))

    seasons_train, seasons_test = sk.model_selection.train_test_split(seasons, test_size=test_size)
    if split_val:
        seasons_val, seasons_test = sk.model_selection.train_test_split(seasons_test, test_size=0.5)
    print("Test seasons: {}".format(seasons_test))

    ind_train = list(season_series[season_series.isin(seasons_train)].index)
    ind_test = list(season_series[season_series.isin(seasons_test)].index)
    X_train = X[ind_train,:]
    X_test = X[ind_test,:]
    Y_train = Y[ind_train]
    Y_test = Y[ind_test]
    if split_val:
        ind_val = list(season_series[season_series.isin(seasons_val)].index)
        X_val = X[ind_val,:]
        Y_val = Y[ind_val]

    print('Percentage of Positive Training Examples: {:.1%}'.format(Y_train.sum()/Y_train.shape[0]))
    print('Percentage of Positive Test Examples: {:.1%}'.format(Y_test.sum()/Y_test.shape[0]))
    if split_val:
        print('Percentage of Positive Val Examples: {:.1%}'.format(Y_val.sum()/Y_val.shape[0]))

    if split_val:
        return X_train, X_val, X_test, Y_train, Y_val, Y_test, ind_train, ind_val, ind_test
    else:
        return X_train, X_test, Y_train, Y_test, ind_train, ind_test

def custom_gridsearch(param_grid, X_train, Y_train, ind_train, X_val, Y_val, ind_val, id_data):
    param_combos = list(sk.model_selection.ParameterGrid(param_grid))

    best_result, best_secondary = None, None
    all_results = []
    for current_params in progressbar.progressbar(param_combos):

        model = xgb.XGBClassifier(use_label_encoder=False,
                                n_jobs=-1,
                                **current_params)
        model.fit(X_train, Y_train, verbose=False, eval_metric=['logloss'])

        current_results = {'param_'+key:value for key,value in current_params.items()}

        predictions, _ = allnba_predict(model.predict_proba(X_train)[:,1], id_data, ind_train, snubfixer=False)
        f1_train = sk.metrics.f1_score(Y_train, predictions)
        predictions, pred_df = allnba_predict(model.predict_proba(X_val)[:,1], id_data, ind_val, snubfixer=False)
        f1_val = sk.metrics.f1_score(Y_val, predictions)
        predictions = model.predict(X_val)
        roc_auc = sk.metrics.roc_auc_score(Y_val, predictions)
        current_results['mean_train_score'] = f1_train
        current_results['mean_val_score'] = f1_val
        current_results['roc_auc'] = roc_auc
        all_results.append(current_results)

        if (best_result == f1_val and roc_auc > best_secondary) or best_result == None or f1_val > best_result:
            best_model = model
            best_params = current_params
            best_result = f1_val
            best_secondary = roc_auc

    print("Best model - accuracy: {}, roc_auc: {}".format(best_result, best_secondary))

    return best_model, all_results, best_params

def rank_season(pred_season):
    for position in ['G', 'F', 'C']:
        pred_season = pred_season.sort_values([position, 'proba'], ascending=[False, False])
        pred_season[position+'_rank'] = (pd.Series(np.arange(len(pred_season)), index=pred_season.index)+1).where(pred_season[position], np.nan)
    poscols = pred_season[['G_rank', 'F_rank', 'C_rank']]
    # make C rank equivalent as only 3 positions per team, now ranks are 1,3,5,7...
    poscols['C_rank'] = poscols['C_rank'] * 2 - 1
    # keep primary position if equal ranks in each eligible position or if the player is one of the top 2 ranks in a position
    pred_season['selected_pos'] = pred_season['pos1'].where((poscols.eq(poscols.min(axis=1),axis=0).sum(axis=1)>1) |
                                                            (poscols.min(axis=1) <= 2),
                                                            poscols.idxmin(axis=1).str.replace('_rank',''))

    return pred_season['selected_pos']

def rank_positions(season_df, pos_field):
    season_df.loc[:,'pos_rank'] = season_df.groupby(pos_field)['proba'].rank('first', ascending=False)
    season_df.loc[:,'y_pred'] = 0
    selected_index = season_df[((season_df[pos_field] == 'C') & (season_df['pos_rank'] <= 3)) |
                       ((season_df[pos_field] != 'C') & (season_df['pos_rank'] <= 6))].index
    season_df.loc[selected_index, 'y_pred'] = 1

    return season_df[['pos_rank', 'y_pred']]

def snub_position(season_df, pos_field, snub_threshold=0.1):
    snub_idx = season_df[season_df['y_pred'] == 0]['proba'].idxmax()
    snub_prob = season_df.loc[snub_idx, 'proba']
    snub_pos = season_df.loc[snub_idx, pos_field]

    worst_idx = season_df[(season_df['y_pred'] == 1) & (season_df['selected_pos'] != snub_pos)]['proba'].idxmin()
    worst_prob = season_df.loc[worst_idx, 'proba']
    worst_pos = season_df.loc[worst_idx, pos_field]

    # check that snub was actually deserving compared to players in other positions
    if snub_prob - worst_prob > snub_threshold:

        positions_list = ['G','F','C']
        snub_positions = list(season_df.loc[snub_idx, positions_list])
        other_positions = [pos for pos, check in zip(positions_list, snub_positions) if check and pos != snub_pos]

        continue_searching = True

        # try switching position for snub
        if len(other_positions) > 0:
            min_other = season_df[(season_df[pos_field] == other_positions[0]) &
                               (season_df['y_pred'] == 1)]['proba'].min()
            if snub_prob > min_other:
                season_df.loc[snub_idx, pos_field] = other_positions[0]
                continue_searching = False

        # try switching another player so snub will move up in its position
        if continue_searching:
            snubpos_selections = season_df[(season_df['y_pred'] == 1) &
                                        (season_df[pos_field] == snub_pos)]
            multipos_players = snubpos_selections.sort_values('proba')[positions_list].sum(axis=1) > 1

            multipos_idx = list(multipos_players[multipos_players].index)

            # loop through potential cantidates, starting with lowest ranked
            for idx in multipos_idx:
                if continue_searching:
                    multipos_prob = season_df.loc[idx, 'proba']
                    multipos_positions = list(season_df.loc[idx, positions_list])
                    switch_position = [pos for pos, check in zip(positions_list, multipos_positions) if check and pos != snub_pos][0]

                    min_other = season_df[(season_df[pos_field] == switch_position) &
                                           (season_df['y_pred'] == 1)]['proba'].min()

                    # only switch if player will remain an allnba player in new position
                    if multipos_prob > min_other:
                        season_df.loc[idx, pos_field] = switch_position
                        continue_searching = False

    return season_df[[pos_field]]

def allnba_predict(proba, id_data, ind_test=[], pos_field='selected_pos', snubfixer=True):
    if ind_test:
        pred_df = id_data.loc[ind_test,:]
    else:
        pred_df = id_data.copy()
    pred_df.loc[:,'proba'] = proba
    pred_df = pred_df.reset_index(drop=True)

    if pos_field == 'selected_pos':
        alternative_pos = pred_df.groupby('season').apply(lambda x: rank_season(x)).reset_index()
        if len(set(pred_df['season'])) > 1:
            alternative_pos = alternative_pos.set_index('level_1').drop(columns=['season'])
        else:
            alternative_pos = alternative_pos.drop(columns=['season']).T.rename(columns={0:'selected_pos'})
        alternative_pos.index.name = None
        pred_df = pred_df.merge(alternative_pos, how='left', left_index=True, right_index=True)

    pred_df[['pos_rank','y_pred']] = pred_df.groupby('season').apply(lambda x: rank_positions(x, pos_field))

    if snubfixer:
        # see if any positions can be changed to select most deserving cantidates
        pred_df.loc[:,'selected_pos'] = pred_df.groupby('season').apply(lambda x: snub_position(x, pos_field)).loc[:,pos_field]
        # rerank after swithcing positions
        pred_df[['pos_rank','y_pred']] = pred_df.groupby('season').apply(lambda x: rank_positions(x, pos_field))

    if 'selected_pos' not in pred_df.columns:
        pred_df.loc[:,'selected_pos'] = pred_df[pos_field]
    # pred_centres = pred_df[pred_df[pos_field] == 'C'].loc[:,['season',pos_field, 'y_pred']]
    # pred_smalls = pred_df[pred_df[pos_field] != 'C'].loc[:,['season',pos_field, 'y_pred']]
    # predictions = np.zeros(y_pred.shape)
    # predictions[list(pred_centres.groupby('season')['y_pred'].nlargest(3).index.levels[1])] = 1
    # predictions[list(pred_smalls.groupby(['season',pos_field])['y_pred'].nlargest(6).index.levels[2])] = 1

    return np.array(pred_df['y_pred']), pred_df

def graph_grid_results(val_results, params=[]):
    results_df = pd.DataFrame(val_results)

    if len(params) == 0:
        params = [x for x in results_df if 'param_' in x]

    fig, (axs) = plt.subplots(1, len(params), sharey=True, figsize=(10,6))

    if len(params) == 2:
        x = list(results_df['param_'+params[0]])
        y = list(results_df['param_'+params[1]])
        for test_train, ax in zip(['train', 'test'], axs):
            z = list(results_df['mean_{}_score'.format(test_train)])

            ax.scatter(x, y, c=z, alpha=0.2, cmap='RdYlGn')
            ax.set_title(test_train)
            ax.set_xlabel(params[0])
            ax.set_ylabel(params[1])
    else:
        for param, ax in zip(params, axs):

            for test_train, plot_colour in zip(['test','val','train'], ['blue','cyan','orange']):
                if 'mean_{}_score'.format(test_train) in results_df.columns:
                    x = list(results_df[param])
                    y = list(results_df['mean_{}_score'.format(test_train)])
                    test_average = results_df.groupby(param)['mean_{}_score'.format(test_train)].mean()
                    x2 = list(test_average.index)
                    y2 = list(test_average)

                    ax.plot(x, y, 'o', c=plot_colour, alpha=0.2)
                    ax.plot(x2, y2, label=test_train, c=plot_colour)

                    ax.set_title(param.replace('param_',''))

        plt.legend();
