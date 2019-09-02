import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_selection import f_regression, SelectKBest, f_classif
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import sys

def reduce_mem_usage(df, verbose=True):
    """ Function iterates through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        Credit to: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
        Parameters
        ----------
        df : Pandas DataFrame
        verbose: (True) by default, prints out before and after memory usage
        Returns
        -------
        df : Reduced Memory Pandas DataFrame
    """

    if verbose:
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(
            100 * (start_mem - end_mem) / start_mem))

    return df


def load_dataset(file_path, verbose=True):
    if not os.path.isfile(file_path):
        raise IOError(f'Invalid {file_path} file path.')

    df = pd.read_csv(file_path)
    df = reduce_mem_usage(df, verbose=verbose)
    if verbose:
        file_name = file_path.split('/')[-1]
        print(
            f'''
        -------------------- SHAPE ---------------------
        DF {file_name}: {df.shape}
        ------------------------------------------------
        ''')
    return df


def save_data_frame(df=None, filename=None):
    """
    Saves data frame to csv format
    Parameters
    ----------
    df: Pandas DataFrame
    filename: File path or name
    Returns
    -------
    csv file
    """
    try:
        if not filename.endswith('.csv'):
            filename += '.csv'
        df.to_csv(filename, index=False)
        print(f"Data Frame saved @:{filename}")
    except Exception as e:
        print("Data Frame couldn't be saved: ", sys.exc_info()[0])
        raise

def correlations(data, y, xs):
    """
    Computes Pearsons and Spearman correlation coefficient.
    Parameters
    ----------
    data: Pandas Data Frame
    y: Target/Dependent variable - has to be python string object
    xs: Features/Independent variables - python list of string objects
    Returns
    ------
    df: Pandas Data Frame Object
    """
    if data is None:
        raise ValueError(
            "The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")
    if (y is None) or (xs is None):
        raise ValueError(
            "The parameter `y` or `xs` has to be non-nil reference")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` - has to be Pandas DataFrame object")
    if not isinstance(y, str):
        raise ValueError("`data` - has to be Python string object")
    if not isinstance(xs, list):
        raise ValueError("`xs` - has to be Python list object")

    rs = []
    rhos = []
    for x in xs:
        r = stats.pearsonr(data[y], data[x])[0]
        rs.append(r)
        rho = stats.spearmanr(data[y], data[x])[0]
        rhos.append(rho)
    return pd.DataFrame({"feature": xs, "r": rs, "rho": rhos})


def hist_plot(df, col_name, title=None, xlabel=None, ylabel='Density'):
    """
    Plot's histogram
    Parameters
    ----------
    df : Pandas Data Frame
    col_name : column name in data frame
    title : Plot title
    xlabel : X-axis label
    ylabel : Y-axis label
    """
    fig = plt.figure(figsize=(10, 6))  # define plot area
    ax = fig.add_subplot(111)  # add single subplot
    sns.distplot(df[col_name], ax=ax)  # Use seaborn plot
    if not title:
        title = 'Histogram of {}'.format(col_name)
    ax.set_title(title)  # Give the plot a main title
    if not xlabel:
        xlabel = col_name
    ax.set_xlabel(xlabel)  # Set text for the x axis
    ax.set_ylabel(ylabel)  # Set text for y axis


def box_plot(df, col_name, title=None, xlabel=None):
    """
    Draw's a single horizontal boxplot
    Parameters
    ----------
    df : Pandas Data Frame
    col_name : column name in data frame
    title : Plot title
    xlabel : X-axis label
    ylabel : Y-axis label
    """
    fig = plt.figure(figsize=(10, 6))  # define plot area
    ax = fig.add_subplot(111)  # add single subplot
    sns.boxplot(df[col_name], ax=ax)  # Use seaborn plot
    if not title:
        title = 'Boxplot of {}'.format(col_name)
    ax.set_title(title)  # Give the plot a main title
    if not xlabel:
        xlabel = col_name
    ax.set_xlabel(xlabel)  # Set text for the x axis


def bar_plot(df, col_name, title=None, xlabel=None, ylabel='Count'):
    """
    Draw's a single bar plot
    Parameters
    ----------
    df : Pandas Data Frame
    col_name : column name in data frame
    title : Plot title
    xlabel : X-axis label
    ylabel : Y-axis label
    """
    fig = plt.figure(figsize=(10, 6))  # define plot area
    ax = fig.add_subplot(111)  # add single subplot
    ax = df[col_name].value_counts().plot.bar(
        color='steelblue')  # Use pandas bar plot
    if not title:
        title = 'Barplot of {}'.format(col_name)
    ax.set_title(title)  # Give the plot a main title
    if not xlabel:
        xlabel = f'No. of {col_name}'
    ax.set_xlabel(xlabel)  # Set text for the x axis
    ax.set_ylabel(ylabel)  # Set text for the y axis


def get_categorical_columns(df, min_card=None, max_card=None):
    """
    Returns categorical columns from pandas dataframe
    Parameters
    ----------
    df : Pandas Dataframe
    min_card : Minimum Cardinality (default = None)
    max_card : Maximum Cardinality (default = None)
    Returns
    -------
    Python list
    """
    if df is None:
        raise ValueError(
            "The parameter 'df' must be assigned a non-nil reference to a Pandas DataFrame")
    cat_cols = df.select_dtypes(
        include=['category', 'object']).columns.tolist()
    if min_card and (max_card is None):
        cat_cols = [col for col in cat_cols if df[col].nunique() >= min_card]
    elif max_card and (min_card is None):
        cat_cols = [col for col in cat_cols if df[col].nunique() <= max_card]
    elif min_card and max_card:
        cat_cols = [col for col in cat_cols if (
            df[col].nunique() >= min_card) and (df[col].nunique() <= max_card)]
    return cat_cols


def get_numeric_columns(df):
    """
    Returns numerical columns from pandas dataframe
    Parameters
    ----------
    df : Pandas Dataframe
    Returns
    -------
    Python list
    """
    if df is None:
        raise ValueError(
            "The parameter 'df' must be assigned a non-nil reference to a Pandas DataFrame")
    return list(df.select_dtypes(exclude=['category', 'object']))


def plot_numerical_columns_reg(df, target_col, alpha=0.5, color='grey'):
    """
    Plots numerical features vs. numerical target
    Parameters
    ----------
    df : Pandas Dataframe
    target_col : Target variable, dependent variable
    """
    num_columns = get_numeric_columns(df)
    for col in sorted(num_columns):
        if col != target_col:
            sns.lmplot(x=col, y=target_col, data=df,
                       scatter_kws=dict(alpha=alpha, color=color))
            plt.title(f'{target_col} vs. {col}')
            plt.show()
            plt.close()


def plot_categorical_reg(df, target_col, min_card=2, max_card=12, height=5, ascpect=2, rotation=45, color='grey', kind='bar'):
    """
    Plots categorical features vs. numerical target
    Parameters
    ----------
    df: Pandas Dataframe
    traget_col: Target variable, dependent variable
    min_card: Minimum cardinality of categorical feature
    max_card: Maximum cardinality of categorical feature
    """
    cat_columns = get_categorical_columns(df)
    for col in sorted(cat_columns):
        if (df[col].nunique() >= min_card) and (df[col].nunique() < max_card):
            sns.catplot(x=col, y=target_col, data=df, kind=kind,
                        color=color, height=height, aspect=ascpect)
            plt.xticks(rotation=rotation)
            plt.title(f'{target_col} vs. {col}')
            plt.show()
            plt.close()


def plot_correlation_heatmap(data=None, vmax=1, annot=True, corr_type='pearson', figsize=(12, 12)):
    """
    Plots correlations on a heatmap
    """
    if data is None:
        raise ValueError(
            "The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")
    # Compute the correlation matrix
    corr = data.corr(corr_type)
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    fig, axes = plt.subplots(figsize=figsize)
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=vmax, annot=annot, square=True,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=axes)
    plt.show()
    plt.close()


def joint_plot(df, col_name1, col_name2, title=None, alpha=0.75):
    """
    Draws joint plot
    """
    sns.jointplot(x=col_name1, y=col_name2, data=df, alpha=alpha)
    if not title:
        title = 'Joint plot of {} vs. {}'.format(col_name1, col_name2)
    plt.suptitle(title)
    plt.show()
    plt.close()


def get_quantiles(df, cols):
    """
    Returns quantiles 1%, 25, 50, 75, 95 and 99%
    """
    return df[cols].quantile(q=[.01, .25, .5, .75, .95, .99]).to_frame()


def get_leq_quantile(df, col, q):
    """
    Returns values less than or equal to passed quantile
    """
    return df[(df[col] <= df[col].quantile(q=q))].to_frame()


def get_geq_quantile(df, col, q):
    """
    Returns values greater than or equal to passed quantile
    """
    return df[(df[col] >= df[col].quantile(q=q))].to_frame()


def get_between_quantiles(df, col, qs):
    """
    Returns values >= and <= passed quantile ranges
    """
    lower_q = min(qs)
    upper_q = max(qs)
    return df[(df[col] >= df[col].quantile(q=lower_q)) & (df[col] <= df[col].quantile(q=upper_q))].to_frame()


def lowess_scatter(data, x, y, jitter=0.0, skip_lowess=False):
    if skip_lowess:
        fit = np.polyfit(data[x], data[y], 1)
        line_x = np.linspace(data[x].min(), data[x].max(), 10)
        line = np.poly1d(fit)
        line_y = list(map(line, line_x))
    else:
        lowess = sm.nonparametric.lowess(data[y], data[x], frac=.3)
        line_x = list(zip(*lowess))[0]
        line_y = list(zip(*lowess))[1]

    figure = plt.figure(figsize=(10, 6))
    axes = figure.add_subplot(1, 1, 1)
    xs = data[x]
    if jitter > 0.0:
        xs = data[x] + stats.norm.rvs(0, 0.5, data[x].size)

    axes.scatter(xs, data[y], marker="o", color="steelblue", alpha=0.5)
    axes.plot(line_x, line_y, color="DarkRed")

    title = "Plot of {0} v. {1}".format(x, y)

    if not skip_lowess:
        title += " with LOESS"
    axes.set_title(title)
    axes.set_xlabel(x)
    axes.set_ylabel(y)

    plt.show()
    plt.close()


def plot_scatter_by_groups(df, x_col, y_col, group_by_col, colors=None, alpha=0.75):
    labels = df[group_by_col].unique()
    num_labels = np.arange(1, len(labels)+1)
    fig, ax = plt.subplots()
    for idx, label in zip(num_labels, labels):
        indices_to_keep = df[group_by_col] == label
        y = df.loc[indices_to_keep, y_col]
        if x_col == 'index':
            x = df.index[indices_to_keep]
        else:
            x = df.loc[indices_to_keep, x_col]
        ax.scatter(x, y, label=label, alpha=alpha)
    plt.show()
    plt.close()


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
    from sklearn.model_selection import train_test_split
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