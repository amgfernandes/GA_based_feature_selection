# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import time
import datetime
import os

'''GA based approach'''
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.callbacks import ProgressBar
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
# %%
dataset = pd.read_csv('data/parkinsons_data.csv')
dataset.columns

# %%
'''data and labels'''
Target = 'status'
y = dataset[Target]

additional_columns_to_drop = 'name'

if additional_columns_to_drop is not None:
    additional_columns=additional_columns_to_drop
    X = dataset.drop(columns=additional_columns_to_drop)
    X = X.drop(columns=[Target])
else:
    X = dataset.drop(columns= [Target])


quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)

'''convert back to dataframes'''
X_train_trans = pd.DataFrame(X_train_trans,columns=X.columns)
X_test_trans = pd.DataFrame(X_test_trans,columns=X.columns)

# %%
clf = GradientBoostingClassifier(n_estimators=10)
clf.fit(X_train_trans, y_train)
y_predict = clf.predict(X_test_trans)
accuracy_no_GA = accuracy_score(y_test, y_predict)
print("accuracy score without GA selection: ", "{:.2}".format(accuracy_no_GA))

# %%
def run_GA(generations, population_size,crossover_probability):

    evolved_estimator = GAFeatureSelectionCV(
        estimator=clf,
        cv=5,
        scoring="accuracy",
        population_size=population_size,
        generations=generations,
        n_jobs=-1,
        crossover_probability=crossover_probability,
        verbose=True,
        max_features=None,
        keep_top_k=3,
        elitism=True
        )

    callback = ProgressBar()
    evolved_estimator.fit(X_train_trans, y_train, callbacks=callback)
    features = evolved_estimator.best_features_
    y_predict_ga = evolved_estimator.predict(X_test_trans.iloc[:,features])
    accuracy = accuracy_score(y_test, y_predict_ga)
    print(evolved_estimator.best_features_)
    print("accuracy score: ", "{:.2}".format(accuracy))
    plt.figure()
    plot_fitness_evolution(evolved_estimator, metric="fitness")
    plt.savefig('fitness.png')

    #print(f'Selected features:', X_test_trans.iloc[:,features].columns)
    selected_features= X_test_trans.iloc[:,features].columns
    cv_results= evolved_estimator.cv_results_
    history= evolved_estimator.history
    return cv_results, history, selected_features

# %%
def main():
    print("Running main function")

    cv_results, history, selected_features =run_GA(generations=generations,
                                population_size=population_size,
                                crossover_probability=crossover_probability)
    results_df= pd.DataFrame(cv_results)
    results_df.to_csv('results_df.csv')

    history_df= pd.DataFrame(history)
    history_df.to_csv('history_df.csv')

    plt.figure()
    sns.violinplot(data=history_df.iloc[:,1:])
    plt.savefig('history_results.png')
    return history_df, selected_features

#%%

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Running GA based feature selection')
    parser.add_argument('--generations', '-g', default=30)
    parser.add_argument('--population_size', '-p',default=30)
    parser.add_argument('--crossover_probability', '-c',default=0.1)
    parser.add_argument('--outdir', help='Location for saving log. Default current directory', default=os.getcwd())
    args = parser.parse_args()
    print(vars(args))

    generations = int(args.generations)
    population_size = int(args.population_size)
    crossover_probability =float(args.crossover_probability)

    RESULTS_DIR = args.outdir

    LOG_FILE = os.path.join(RESULTS_DIR, f'log.txt')
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(LOG_FILE), 
                        logging.StreamHandler()])

    start_time = time.time()
    hr_start_time = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"starting time: {hr_start_time}")
    history_df, selected_features = main()
    TOTAL_TIME = f'Total time required: {time.time() - start_time} seconds'
    hr_end_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"Number of generations: {generations}")
    logging.info(f"Population_size: { population_size}")
    logging.info(f"Crossover_probability: {crossover_probability}")
    logging.info(f"Max accuracy with all features: {accuracy_no_GA}")
    logging.info(f"End time: {hr_end_time}")
    logging.info(f"Max fitness with selection: {history_df.fitness.max()}")
    logging.info(f"Selected features:, {selected_features}")
    logging.info(TOTAL_TIME)
    logging.info('done!')
