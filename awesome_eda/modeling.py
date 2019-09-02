import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_selection import f_regression, SelectKBest, f_classif
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split


def subset_features(train, val, test, features, target):
    X_train = train[features]
    y_train = train[target]
    X_val = val[features]
    y_val = val[target]
    X_test = test[features]
    try:
        y_test = test[target]
    except:
        y_test = None
    return X_train, X_val, X_test, y_train, y_val, y_test



def print_shape(train, val, test=None, title='Train/Val'):
    print('------- SHAPE - {} ---------'.format(title))
    print(f'Training Set: {train.shape}')
    print(f'Validation Set: {val.shape}')
    if test is not None:
        print(f'Testing Set: {test.shape}')




def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=42):
    """
    Function does 3-Way hold out splot of Train, Validate, Test
    """
    train, test = train_test_split(df, test_size=0.2, random_state=seed)
    train, validate = train_test_split(
        train, test_size=0.25, random_state=seed)
    return (train, validate, test)


def get_rmse_mae_r2(y_true, y_pred, verbose=False, title='TEST'):
    """
    Returns regression metrics like - R^2, MSE, RMSE, and MAE as dictionary,
    and prints them out if verbose is set to True.
    Parameters
    ----------
    y_true : Python list or numpy 1D array
    y_pred : Python list or numpy 1D array
    Returns
    -------
    results : Python dictionary
    """
    if (y_true is None) or (y_pred is None):
        raise ValueError(
            'Parameters `y_true` and `y_pred` must be a non-nil reference to python list or numpy arrary')

    assert y_true.shape == y_pred.shape
    # Results dict
    results = {}
    mae_ = mean_absolute_error(y_true, y_pred)
    mse_ = mean_squared_error(y_true, y_pred)
    rmse_ = np.sqrt(mse_)
    r_square_ = r2_score(y_true, y_pred)
    # Store scores
    results['R_SQUARE'] = r_square_
    results['MAE'] = mae_
    results['MSE'] = mse_
    results['RMSE'] = rmse_
    if verbose:
        print(f'-------- {title} SET --------')
        print(f'R^2: {r_square_:.4f}')
        print(f'MSE: {mse_:.4f}')
        print(f'RMSE: {rmse_:.4f}')
        print(f'MAE: {mae_:.4f}')
    return results


def get_reg_metrics(model, X_train, X_val, y_train, y_val, verbose=False):
    """
    Returns a result dictionary containing both training and validation metrics for regression model. 
    If verbose is set True, prints out the results.
    """
    if not model:
        raise ValueError("model has to be a non-nil parameter")

    assert X_train.shape[1] == X_val.shape[1]

    # Results dict
    results = {}
    # Compute Training accuracy
    y_true = y_train
    y_pred = model.predict(X_train)
    results['Train'] = get_rmse_mae_r2(
        y_true, y_pred, verbose=verbose, title='TRAINING')

    # Validation Accuracy
    y_true = y_val
    y_pred = model.predict(X_val)
    results['Validation'] = get_rmse_mae_r2(
        y_true, y_pred, verbose=verbose, title='VALIDATION')
    return results


def find_k_best_features(model, X_train, X_val, y_train, y_val):
    """
    Returns a pandas dataframe with incrementing K features and their metrics for either regeression task or classification task.
    Parameters
    ----------
    model : sklearn model
    X_train : Pandas Dataframe of training set features
    X_val : Pandas Dataframe of validation set features
    y_train : Pandas Series of target column from training set
    y_val : Pandas Series of target column from validation set
    model_type : Default is 'reg' for Regression, change to 'clf' for Classification task
    Returns
    -------
    results : Pandas DataFrame containing K best features and their metrics
    """
    if not model:
        raise ValueError("model has to be a non-nil parameter")

    # Make sure columns match for both training set and validation set
    assert X_train.shape[1] == X_val.shape[1]

    # Get model type
    model_type = getattr(model, "_estimator_type", None)

    # Store results
    results = {}
    results['K'] = []

    if model_type == 'regressor':
        results['R_SQUARE'] = []
        results['MSE'] = []
        results['RMSE'] = []
        results['MAE'] = []
    elif model_type == 'classifier':
        results['ACCURACY'] = []
        results['F1'] = []
    else:
        raise ValueError(
            "Incorrect option was selected for `model_type`, can only be 'reg' or 'clf'")

    # Loop through all the columns
    for k in range(1, len(X_train.columns)+1):
        # Store k num
        results['K'].append(k)

        # Regression task
        if model_type == 'regressor':
            # Select k feature from training set
            selector = SelectKBest(score_func=f_regression, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_val_selected = selector.transform(X_val)
            # Get predicted values
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_val_selected)
            # Compute metrics
            metrics = get_rmse_mae_r2(y_val, y_pred)
            # Store metrics
            results['R_SQUARE'].append(metrics['R_SQUARE'])
            results['MSE'].append(metrics['MSE'])
            results['RMSE'].append(metrics['RMSE'])
            results['MAE'].append(metrics['MAE'])

        # Classification task
        elif model_type == 'classifier':
            # Select k feature from training set
            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_val_selected = selector.transform(X_val)
            # Get predicted values
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_val_selected)
            # Compute metrics
            metrics = get_acc_rec_fone(y_val, y_pred)
            # Store metrics
            results['ACCURACY'].append(metrics['ACCURACY'])
            results['F1'].append(metrics['F1'])
        else:
            raise ValueError('This logic shouldn\'t have been executed')

    if model_type == 'regressor':
        return pd.DataFrame(data={
            'K': results['K'],
            'R_SQUARE': results['R_SQUARE'],
            'MAE': results['MAE'],
            'MSE': results['MSE'],
            'RMSE': results['RMSE']
        })
    elif model_type == 'classifier':
        return pd.DataFrame(data={
            'K': results['K'],
            'ACCURACY': results['ACCURACY'],
            'F1': results['F1']
        })
    else:
        return None