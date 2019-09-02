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








