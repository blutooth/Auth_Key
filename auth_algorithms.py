# Import any required libraries or modules.
import numpy as np
import get_data
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Implement template building here.  Feel free to write any helper classes or functions required.
# Return the generated template for that user.
## templates is a dictionary for user classifiers
"""Creates features, runs classifier, generates submission."""

# defined ensemble class, where predictions are a weighted average of predictors
class EnsembleClassifier():

    # Initialise with classifiers and their weights
    def __init__(self, classifiers,weights):
        self.classifiers = classifiers
        self.weights     = weights

    # Weighted average prediction
    def predict_proba(self, Xs):
        self.predictions_ = list()

        for classifier,X in zip(self.classifiers,Xs):
            self.predictions_.append(classifier.predict_proba(X))
        return np.average(self.predictions_,weights=self.weights, axis=0)


def train_test(train,test):
    # get Train and Test data for the two classifiers respectively
    X_train_efc = get_data.get_data(train)
    X_test_efc  = get_data.get_data(test)

    X_train_rfc =get_data.get_data(train,drop=False)
    X_test_rfc  = get_data.get_data(test,drop=False)

    # classifier and input list to be ensembled
    data_sets=[(X_train_rfc,X_test_rfc,'rfc'),(X_train_efc,X_test_efc,'efc')]

    # templates- classifiers; instances -test X data; answers test y data for user
    print('creating user templates/classifiers')
    templates = dict()
    instances = dict()
    answers   = dict()
    users     = X_train_rfc['non_touch','user'].unique()

    # Create  templates, instances and answers for each user
    for user in users:

        trained_models_user = []  # list of models to be ensembled for each  user
        X_test_user         = []  # list of test data for each user, corresponding to each model

        # create the ensemble
        for (train,test,model_name) in data_sets:

            # Preprocess the training and test data
            print('template for user', user)
            X_train = get_data.map_user_y(train,user)
            Y_train = X_train.pop(('non_touch','user'))
            X_test  = get_data.map_user_y(test,user)
            Y_test  = X_test.pop(('non_touch','user'))

            # Model settings for each ensemble
            if model_name =='rfc':
                model = RandomForestClassifier(n_estimators=1000, max_depth=3, n_jobs=-1,class_weight={1:4,0:1},max_features=None,criterion='entropy')

            elif model_name =='efc':
                model = ExtraTreesClassifier(n_estimators=1500, max_depth=6, n_jobs=-1, min_samples_split=1,class_weight={1:4,0:1},max_features=None,criterion='gini')

            # train each model in the ensemble
            model.fit(X_train,Y_train)
            trained_models_user.append(model)
            # add the ensemble test data
            X_test_user.append(np.array(X_test))

        # return Y_test, X_test and classifiers for each user
        answers[user]=Y_test
        instances[user]=X_test_user
        templates[user]=EnsembleClassifier(trained_models_user,[1,2])

    return (templates,instances,users,answers)

# Implement authentication method here.  Feel free to write any helper classes or functions required.
# Return the authtication score and threshold above which you consider it being a correct user.


def authenticate(instance, user, templates):

    # get model
    model =templates[user]

    # score data.
    score = model.predict_proba(instance)

    # Return score and threshold.
    return score, 0.40

