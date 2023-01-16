from sklearn.datasets import make_classification
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score, balanced_accuracy_score
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from joblib import dump, load
import json
import sys
import os
from sklearn.model_selection import validation_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn_evaluation import plot
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def pooled_var(stds):
    # https://en.wikipedia.org/wiki/Pooled_variance#Pooled_standard_deviation
    n = 5 # size of each group
    return np.sqrt(sum((n-1)*(stds**2))/ len(stds)*(n-1))

def run_experiment(path,tcol,direc,run="model",bool_col=[],scale="false",hyper="random"):

    pdf = pd.read_csv(path)
    X, y = pdf[pdf.columns.difference([tcol])], pdf[tcol]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    if scale == "true":
        print ("APPLYING STANDARD SCALING")
        sc = StandardScaler()
        sc.fit(X_train[X_train.columns.difference(bool_col)])
        sc.fit(X_test[X_test.columns.difference(bool_col)])
        X_train[X_train.columns.difference(bool_col)] = sc.transform(X_train[X_train.columns.difference(bool_col)])
        X_test[X_test.columns.difference(bool_col)] = sc.transform(X_test[X_test.columns.difference(bool_col)])
    clf1 = SVC(random_state=42)
    clf2 = DecisionTreeClassifier()
    clf3 = KNeighborsClassifier()
    clf4 = MLPClassifier(max_iter=3000)
    clf5 = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), random_state=43)
    d = X_train.shape[1]
    metrics = {
            'model':[],
            'params': [],
            'f1-score':[],
            'balanced-accuracy-score':[]
    }
    allmodels = [
        {
            'name': "ANN",
            'classifier': clf4,
            'param_grid': {
                'alpha': [10 ** -x for x in np.arange(-1, 6, 0.5)],
                'hidden_layer_sizes': [(h,) * l for l in [1, 2, 3] for h in [d, d // 2, d * 2]],
                'activation': ['logistic', 'tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'learning_rate_init': sorted([(2**x)/1000 for x in range(8)]+[0.000001])
            },
            'scoring' : 'neg_log_loss'
        },
        {
            'name': "DT",
            'classifier': clf2,
            'param_grid':{
                'criterion': ["gini","entropy"],
                'splitter': ["best","random"],
                'max_depth': np.arange(1, 11, 1),
                'ccp_alpha': clf2.cost_complexity_pruning_path(X_train, y_train)['ccp_alphas']
            },
            'scoring': 'f1'
        },
        {
            "name": "KNN",
            "classifier": clf3,
            "param_grid":{
                "metric": ['manhattan', 'euclidean', 'chebyshev'],
                "n_neighbors": np.arange(1, 51, 3),
                "weights": ['uniform',"distance"],
                "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
            },
            'scoring': 'f1'
        },
        {
            "name": "SVM",
            'classifier': clf1,
            'param_grid':{
                'C': np.logspace(-3,2,6),
                'kernel': ['rbf','linear','sigmoid','poly'],
                'gamma': np.logspace(-3,2,6),
            },
            'scoring': 'f1'
        },
        {
            'name': "AdaBoost",
            'classifier': clf5,
            'param_grid': {
                'base_estimator__max_depth': np.arange(1, 11, 1),
                'base_estimator__splitter': ['random','best'],
                'n_estimators': [1, 2, 5, 10, 20, 30, 45, 60, 80, 90, 100],
                'learning_rate': [(2**x)/100 for x in range(7)]+[1],
            },
            'scoring' : 'f1'
        }
    ]   

    for a in allmodels:
        if run == "model":
            if hyper == "random":
                rs = RandomizedSearchCV(
                    estimator=a['classifier'],
                    param_distributions = a['param_grid'],
                    cv = 5,
                    random_state = 34,
                    refit = True,
                    scoring = a['scoring'],
                    n_jobs = -1,
                    n_iter = 70,
                    return_train_score = True
                )
            else:
                rs = GridSearchCV(
                    estimator=a['classifier'],
                    param_grid = a['param_grid'],
                    cv = 5,
                    refit = True,
                    scoring = a['scoring'],
                    n_jobs = -1,
                    return_train_score = True
                )
            rs.fit(X_train, y_train)
            CHECK_FOLDER = os.path.isdir(direc + "/" + a['name'])
            if not CHECK_FOLDER:
                os.makedirs(direc + "/" + a['name'])
                print("created folder : ", direc + "/" + a['name'])
            dump(rs,direc + "/" + a['name'] + "/" + a['name'] + ".pkl")
        elif run == "validation":
            rs = load(direc + "/" + a['name'] + "/" + a['name'] + ".pkl")
            df = pd.DataFrame(rs.cv_results_)
            results = ['mean_test_score',
           'mean_train_score',
           'std_test_score',
           'std_train_score']


            for idx, (param_name, param_range) in enumerate(a['param_grid'].items()):
                fig,axes = plt.subplots(figsize= (10*len(a['param_grid']),20))
                axes.grid(color='black',linewidth=1.5)
                axes.set_ylabel(a['scoring'], fontsize=40)
                grouped_df = df.groupby(f'param_{param_name}')[results]\
                        .agg({'mean_train_score': 'mean',
                            'mean_test_score': 'mean',
                            'std_train_score': pooled_var,
                            'std_test_score': pooled_var})
                previous_group = df.groupby(f'param_{param_name}')[results]

                if len(grouped_df.index.values) == len(param_range):
                    pr = param_range

                else:
                    pr = grouped_df.index.values
                
                if(grouped_df.index.dtype == np.float64):
                    axes.set_xscale('log')
                    prange = pr
                    axes.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
                else:
                    prange = [str(i) for i in pr]
                axes.set_xlabel(param_name, fontsize=40)
                axes.set_ylim(min(min(grouped_df['mean_train_score'].values),min(grouped_df['mean_test_score'].values)), 1.1)
                axes.tick_params(axis='both',labelsize=25)
                lw = 2
       
                axes.plot(prange, grouped_df['mean_train_score'], label="Training score",color="darkorange", lw=lw)
                axes.plot(prange, grouped_df['mean_test_score'], label="Cross-validation score",color="navy", lw=lw)
                fig.suptitle(a["name"] + "-" + "Validation Curve " + param_name,fontsize=40)
                handles, labels = axes.get_legend_handles_labels()
                axes.legend(handles[::-1],labels[::-1])
                plt.legend(fontsize=30)
                fig.legend(handles, labels, loc=8, ncol=2, fontsize=30)
                plt.savefig(direc + "/" + a['name'] + "/" + a['name'] + "-" + run + "-" + param_name + ".png")
                
        elif run == "validation-num":
            rs = load(direc + "/" + a['name'] + "/" + a['name'] + ".pkl")
            for idx, (param_name, param_range) in enumerate(a['param_grid'].items()):
                try:
                    print (param_name)
                    train_scores, test_scores = validation_curve(
                        rs.best_estimator_, X_train, y_train, param_name=param_name,
                        param_range=param_range,
                        cv=5,
                        scoring=a['scoring'],
                        n_jobs=-1)
                    plot.validation_curve(train_scores, test_scores, param_range, param_name,semilogx=True)
                    plt.savefig(direc + "/" + a['name'] + "/" + a['name'] + run + param_name + ".png")
                    plt.close()
                except:
                    continue
                    
        elif run == "learning":

            rs = load(direc + "/" + a['name'] + "/" + a['name'] + ".pkl")
            cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
            estimator = rs.best_estimator_
            title = "Learning Curves " + a["name"]
            plot_learning_curve(
                    estimator,
                    title,
                    X,
                    y,
                    axes=None,
                    ylim=(0.0, 1.1),
                    cv=cv,
                    n_jobs=8,
                    scoring="balanced_accuracy",
            )
            plt.savefig(direc + "/" + a['name'] + "/" + a['name'] + run + ".png")
        elif run == "test":
            rs = load(direc + "/" + a['name'] + "/" + a['name'] + ".pkl")
            metrics['model'].append(a['name'])
            metrics['params'].append(rs.best_params_)
            estimator = rs.best_estimator_
            fp = estimator.predict(X_test)
            f1 = f1_score(y_test,fp)
            ba = balanced_accuracy_score(y_test,fp)
            metrics['f1-score'].append(f1)
            metrics['balanced-accuracy-score'].append(ba)
            color = 'white'
            matrix = plot_confusion_matrix(estimator, X_test, y_test, cmap=plt.cm.Blues)
            matrix.ax_.set_title('Confusion Matrix', color=color)
            plt.xlabel('Predicted Label', color=color)
            plt.ylabel('True Label', color=color)
            plt.gcf().axes[0].tick_params(colors=color)
            plt.gcf().axes[1].tick_params(colors=color)
            plt.savefig(direc + "/" + a['name'] + "/" + a['name'] + run + ".png")

    if run == "test":
        print ("I AM HERE")
        md = pd.DataFrame.from_dict(metrics)
        md.to_csv(direc + "/" + 'metrics.csv',index=False)


if __name__== "__main__":

    run_experiment(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5].split(","),sys.argv[6],sys.argv[7])
