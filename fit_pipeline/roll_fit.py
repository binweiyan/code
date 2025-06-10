import xgboost as xgb

a = 0
def rolling_fit_score(stock_df, V_return, V_target, train_window_size = 5, unit= 22, lambda_ = 0.1, pred_date = datetime(2020, 1, 1), scales = None, level = 3,weight = True, get_model = xgb.XGBRegressor, save_coefs = False, model_type = 'tree'):

    #divide the training and testing into chunks of days of window_size
    days = stock_df['D'].values
    day_chunks = [days[i * unit:(i + 1) * unit] for i in range((len(days) + unit - 1) // unit)]
    day_chunks_training = []
    for i, day_chunk in enumerate(day_chunks):
        #choose the last train_window_size days as training and the last test_window_size days as testing
        start_date = max(i - train_window_size, 0)
        str_type = "%Y%m%d"
        if '-' in day_chunk[0]:
            str_type = "%Y-%m-%d"
        if datetime.strptime(day_chunk[0], str_type) >= pred_date and i != 0:
            day_chunks_training.append(([x for y in day_chunks[start_date:i] for x in y ],i))
    bigFR_score = []
    # Find the position of dimension "V"
    # dims_list = list(stock_df.dims)  # Convert dimension keys to a list
    # level = dims_list.index("V")  # Find the index of "V"
    print(level)
    coefs = []
    for target in V_target:
        coefs_result = []
        model = get_model()
        #rolling fit and score
        predicts = []
        ground_truth = []
        for day_chunk_training,i in day_chunks_training:
            global X1
            # print(target)
            X1 = stock_df.sel(D = day_chunk_training, V = V_return + [target]).compute().to_dataframe().unstack(level = level)['data_variable'].fillna(0).replace([np.inf, -np.inf], 0)
            X2 = stock_df.sel(D = day_chunks[i], V = V_return + [target]).compute().to_dataframe().unstack(level = level)['data_variable'].fillna(0).replace([np.inf, -np.inf], 0)
            if scales is not None:
                # print(X1)
                if (X1.std() == 0).any() or (X2.std() == 0).any():
                    continue
                if weight:
                    model.fit(X1[V_return], X1[target], sample_weight = (X1[target] != 0))
                else:
                    model.fit(X1[V_return], X1[target])
                aligned_multiplier = X2.index.get_level_values('S').map(scales[target])
                if save_coefs:
                    if model_type == 'tree':
                        coefs_result.append(xr.DataArray(model.feature_importances_.reshape(1, -1, 1), dims = ['D', 'V', 'V2'], coords = {'V': V_return, 'V2': [target], 'D': [day_chunks[i][0]]}))
                    if model_type == 'linear':
                        coefs_result.append(xr.DataArray(model.coef_.reshape(1, -1, 1), dims = ['D', 'V', 'V2'], coords = {'V': V_return, 'V2': [target], 'D': [day_chunks[i][0]]}))
                bigFR_score.append(bigFR(model.predict(X2[V_return]), X2[target] * aligned_multiplier))

            else:

                if (X1.std() == 0).any() or (X2.std() == 0).any():
                    continue

                # normalize the data by V axis
                scale = X2.std()
                X1 = X1 / X1.std()
                X2 = X2 / X2.std()
                # print(scale, X1.max(axis=0), X1.min(axis=0))
                #if X1 > upper_limit, set it to upper_limit
                X1[X1 > upper_limit] = upper_limit
                X1[X1 < -upper_limit] = -upper_limit
                #plot distribution of X1
                # plt.hist(X1[V_return].values.flatten(), bins = 100)
                # plt.show()
                #weight is 1 if target is not 0, otherwise 0
                if weight:
                    model.fit(X1[V_return], X1[target], sample_weight = (X1[target] != 0))
                else:
                    model.fit(X1[V_return], X1[target])
                model.fit(X1[V_return], X1[target])
                if save_coefs:
                    #if tree, save the feature importance
                    #if linear model, save the coefficients
                    if model_type == 'tree':
                        coefs_result.append(xr.DataArray(model.feature_importances_.reshape(1, -1, 1), dims = ['D', 'V', 'V2'], coords = {'V': V_return, 'V2': [target], 'D': [day_chunks[i][0]]}))
                    if model_type == 'linear':
                        coefs_result.append(xr.DataArray(model.coef_.reshape(1, -1, 1), dims = ['D', 'V', 'V2'], coords = {'V': V_return, 'V2': [target], 'D': [day_chunks[i][0]]}))
                bigFR_score.append(bigFR(model.predict(X2[V_return]), X2[target] * scale[target]))

            predicts.append(model.predict(X2[V_return]))
            ground_truth.append(X2[target])
        print(np.corrcoef(np.concatenate(predicts), np.concatenate(ground_truth))[0][1], target)
        #plot the score of each target
        plt.plot(np.cumsum(bigFR_score), label = target)
        bigFR_score = []
        if save_coefs:
            if len(coefs_result) == 0:
                continue
            coefs.append(xr.concat(coefs_result, dim = 'D'))

        #scales

    plt.legend()
    plt.show()
    #construct xarray of the coefficients, D is the start of prediction date (day_chunks[i][0])
    if len(coefs) == 0:
        return None
    if save_coefs:
        coefs = xr.concat(coefs, dim = 'V2')    
    return coefs