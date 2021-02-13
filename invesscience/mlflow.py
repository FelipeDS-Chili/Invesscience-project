import multiprocessing
import time
import warnings
import category_encoders as ce
import joblib
import mlflow
import pandas as pd
import os
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from psutil import virtual_memory
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from termcolor import colored
from xgboost import XGBRegressor
from invesscience.utils import compute_f1, simple_time_tracker, compute_precision, get_data_filled
from invesscience.joanna_merge import get_training_data
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer,SimpleImputer

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler,StandardScaler

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier , GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from scipy.stats import uniform, randint
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import Binarizer
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as Pipeline_imb
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB



MLFLOW_URI = "https://mlflow.lewagon.co/"


class Trainer(object):
    ESTIMATOR = "LogisticRegression"
    EXPERIMENT_NAME = "Invesscience_batch_#463"
    IMPUTER = 'SimpleImputer'
    SCALER_AMOUNT = 'RobustScaler'
    SCALER_PROFESSIONALS = 'MinMaxScaler'
    SCALER_TIME = 'StandardScaler'
    SCALER_PARTICIPANTS = 'StandardScaler'

    def __init__(self, X, y, **kwargs):
        """
        FYI:
        __init__ is called every time you instatiate Trainer
        Consider kwargs as a dict containig all possible parameters given to your constructor
        Example:
            TT = Trainer(nrows=1000, estimator="Linear")
               ==> kwargs = {"nrows": 1000,
                            "estimator": "Linear"}
        :param X:
        :param y:
        :param kwargs:
        """

        self.pipeline = None
        self.kwargs = kwargs
        self.estimator_xgb = self.kwargs.get("estimator", self.ESTIMATOR)
        self.local = kwargs.get("local", False)  # if True training is done locally
        self.year = kwargs.get("year", '2014')
        self.smote = kwargs.get("smote", False)
        self.mlflow = kwargs.get("mlflow", False)
        self.reference = kwargs.get("reference", 'a')
        self.tag = kwargs.get("tag_description", "nada")
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME)  # cf doc above
        self.model_params = None  # for
        self.grid_search_choice = kwargs.get("grid_search_choice", False)
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True)  # cf doc above
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                                  test_size=0.3)
        self.nrows = self.X_train.shape[0]  # nb of rows to train on
        self.log_kwargs_params()
        self.log_machine_specs()

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        if estimator == "LogisticRegression":
            model = LogisticRegression(class_weight= 'balanced')
        elif estimator == "SVC":
            model = SVC(class_weight='balanced')
        elif estimator == "KNeighborsClassifier":
            model = KNeighborsClassifier()
        elif estimator == "DecisionTree":
            model = DecisionTreeClassifier(class_weight ='balanced')

        elif estimator == "RandomForestClassifier":
            model = RandomForestClassifier(bootstrap=True, ccp_alpha=4.795609695177735,
                                                     criterion='entropy', max_depth=4,
                                            max_features='sqrt', max_samples=0.21073053471890313, n_estimators=254)
            self.model_params = {  # 'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
                'max_features': ['auto']}
            # 'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}

        elif estimator == "xgboost":

            model = XGBClassifier()

        elif estimator == "GaussianNB":

            model = GaussianNB()

        elif estimator == "adaboost":
            model = AdaBoostClassifier()

        elif estimator =='voting':

            model_1 = SGDClassifier(alpha=0.30172076992185237, class_weight='balanced', early_stopping=True,
                        eta0=0.0001, learning_rate='constant', loss='log', n_iter_no_change=10, validation_fraction=0.3)
            model_2 = SGDClassifier(alpha=0.5471642701893039, class_weight='balanced', early_stopping=True,
                         eta0=0.0001, loss='modified_huber', n_iter_no_change=10, validation_fraction=0.3)

            model_3 = SGDClassifier(alpha=0.3566042451846202, class_weight='balanced', early_stopping=True,
                                     eta0=0.0001, loss='modified_huber', n_iter_no_change=10, validation_fraction=0.3)

            model_4 = SGDClassifier(alpha=0.14886018175436921, class_weight='balanced', early_stopping=True,
                                eta0=0.0001, loss='modified_huber', n_iter_no_change=10, validation_fraction=0.3)
            model_5 = SVC(C=7.661584366688032, class_weight='balanced', degree=1, gamma=2.8796293228415792, kernel='poly', probability =True)
            model_6 = SVC(C=6.359827845516727, class_weight='balanced', gamma=4.6924698110083485, kernel='linear', probability =True)



            model = VotingClassifier(estimators=[('model1a', model_1),
                                                ('model2a', model_2),
                                                ('model4a', model_4),
                                                ('model5a', model_5),
                                                ('model6a', model_6)]
                                ,voting='soft')




#[3, 1, 3, 2, 3]
#[6, 2, 5, 3, 4]
#[7, 4, 5, 3, 5]


        elif estimator =='SGDC':
            model = SGDClassifier(class_weight ='balanced')

#[1,2,3,4,5,6] --> 0,454
#[5,6,2,1,6,4] --> 0,361
#[8,7,6,5,4,3]
#[6,5,7,4,8,3]


        #else:
            #model = Lasso()


        estimator_params = self.kwargs.get("estimator_params", {}) #Dictionary
        self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, "red"))
        return model

    def get_imputer(self):
        imputer = self.kwargs.get("imputer", self.IMPUTER)
        if imputer == "SimpleImputer":
            imputer_use = SimpleImputer()
        if imputer == "KNNImputer":
            imputer_use = KNNImputer()


        imputer_params = self.kwargs.get("imputer_params", {})
        self.mlflow_log_param("imputer", imputer)
        imputer_use.set_params(**imputer_params)
        print(colored(imputer_use.__class__.__name__, "blue"))

        return imputer_use



    def get_scaler_raised_amount(self):
        scaler_amount = self.kwargs.get("scaler_amount", self.SCALER_AMOUNT)
        if scaler_amount == "RobustScaler":
            scaler_use = RobustScaler()
        elif scaler_amount == "StandardScaler":
            scaler_use = StandardScaler()
        elif scaler_amount == 'MinMaxScaler':
            scaler_use = MinMaxScaler()

        scaler_amount_params = self.kwargs.get("scaler_amount_params", {})
        self.mlflow_log_param("scaler_amount", scaler_amount)
        scaler_use.set_params(**scaler_amount_params)
        print(colored(scaler_use.__class__.__name__, "blue"))

        return scaler_use


    def get_scaler_professionals(self):
        scaler_professionals = self.kwargs.get("scaler_professionals", self.SCALER_PROFESSIONALS)
        if scaler_professionals == "RobustScaler":
            scaler_use = RobustScaler()
        elif scaler_professionals == "StandardScaler":
            scaler_use = StandardScaler()
        elif scaler_professionals == 'MinMaxScaler':
            scaler_use = MinMaxScaler()

        scaler_professionals_params = self.kwargs.get("scaler_professionals_params", {})
        self.mlflow_log_param("scaler_professionals", scaler_professionals)
        scaler_use.set_params(**scaler_professionals_params)
        print(colored(scaler_use.__class__.__name__, "blue"))

        return scaler_use

    def get_scaler_time(self):
        scaler_time = self.kwargs.get("scaler_time", self.SCALER_TIME)
        if scaler_time == "RobustScaler":
            scaler_use = RobustScaler()
        elif scaler_time == "StandardScaler":
            scaler_use = StandardScaler()
        elif scaler_time == 'MinMaxScaler':
            scaler_use = MinMaxScaler()

        scaler_time_params = self.kwargs.get("scaler_time_params", {})
        self.mlflow_log_param("scaler_time", scaler_time)
        scaler_use.set_params(**scaler_time_params)
        print(colored(scaler_use.__class__.__name__, "blue"))

        return scaler_use


    def get_scaler_participant(self):
        scaler_participants = self.kwargs.get("scaler_participants", self.SCALER_PARTICIPANTS)
        if scaler_participants == "RobustScaler":
            scaler_use = RobustScaler()
        elif scaler_participants == "StandardScaler":
            scaler_use = StandardScaler()
        elif scaler_participants == 'MinMaxScaler':
            scaler_use = MinMaxScaler()

        scaler_participant_params = self.kwargs.get("scaler_participant_params", {})
        self.mlflow_log_param("scaler_participants", scaler_participants)
        scaler_use.set_params(**scaler_participant_params)
        print(colored(scaler_use.__class__.__name__, "blue"))

        return scaler_use


    def set_pipeline(self):

        if self.reference=='a':

            if self.year == '2009':
                categorical_features_1 = ['category_code', 'country_code','state_code', 'founded_at','timediff_founded_series_a' ,'time_diff_series_a_now']

            elif self.year == '2014':
                categorical_features_1 = ['category_code', 'country_code','state_code', 'founded_at','timediff_founded_series_a', 'time_diff_series_a_now'] #first use imputer /after ohe


            categorical_features_2 = ['participants_a', 'raised_amount_usd_a', 'rounds_before_a', 'mean_comp_worked_before',  'founder_count', 'degree_count'] # impute first, after ordinals

            booleans_features = ['graduate', 'undergrad','professional', 'MBA_bool', 'cs_bool', 'phd_bool', 'top_20_bool', 'mean_comp_founded_before', 'female_ratio'] # ordinals/binaries




            #Defining imputers
            imputer = self.get_imputer()
            imputer_2= SimpleImputer(strategy = 'most_frequent')

            #pipes for each feature

            pipe_1 = Pipeline([('imputer', imputer_2),
                                ('ohe', OneHotEncoder(handle_unknown='ignore'))])

            pipe_2 = Pipeline([('imputer_ord', imputer),
                                    ('ord_encoder', OneHotEncoder(handle_unknown='ignore'))
                               ])

            pipe_bool =  Pipeline([('imputer_bool', imputer_2),
                                    ('ord_encoder', OneHotEncoder(handle_unknown='ignore'))
                               ])

            #process

            feateng_blocks = [ ('cat_ohe', pipe_1, categorical_features_1),
                               ('cat_ord',  pipe_2, categorical_features_2),
                               ('cat_bool',  pipe_bool, booleans_features)
                               ]



        elif self.reference ==0:

            if self.year == '2014':
                categorical_features_1 = ['category_code', 'country_code','state_code', 'founded_at','timediff_founded_series_0', 'time_diff_series_0_now' ] #first use imputer /after ohe


            categorical_features_2 = ['participants_0' , 'raised_amount_usd_0','participants_0', 'mean_comp_worked_before','founder_count', 'degree_count'] # impute first, after ordinals

            booleans_features = ['graduate', 'undergrad', 'professional',  'MBA_bool', 'cs_bool', 'phd_bool','top_20_bool', 'mean_comp_founded_before','female_ratio'] # ordinals/binaries


            #Defining imputers
            imputer = self.get_imputer()
            imputer_2 = SimpleImputer(strategy = 'most_frequent')

            #pipes for each feature

            pipe_1 = Pipeline([('imputer', imputer_2),
                                ('ohe', OneHotEncoder(handle_unknown='ignore'))])

            pipe_2 = Pipeline([('imputer_ord', imputer),
                                    ('ord_encoder', OneHotEncoder(handle_unknown='ignore'))
                               ])

            pipe_bool =  Pipeline([('imputer_bool', imputer_2),
                                    ('ord_encoder', OneHotEncoder(handle_unknown='ignore'))
                               ])
            #process

            feateng_blocks = [ ('cat_1', pipe_1, categorical_features_1),
                               ('cat_2',  pipe_2, categorical_features_2),
                               ('cat_bool', pipe_bool, booleans_features)]




        #Columntransformer keeping order
        preprocessor = ColumnTransformer(feateng_blocks, remainder= 'passthrough')

        #final_pipeline
        self.pipeline = Pipeline(steps = [('preprocessing', preprocessor),
                            ('model_use', self.get_estimator())] )




        if self.smote:

            smote =ADASYN(sampling_strategy = 'minority', n_neighbors= 20)
            self.pipeline =Pipeline_imb([
                ('prep',preprocessor),
                ('smote', smote),
                ('model_use', self.get_estimator())])


        # Random search
        if self.grid_search_choice:
            grid_search = RandomizedSearchCV(
                self.pipeline,
                param_distributions ={

                "model_use__learning_rate": uniform(0,1),
               "model_use__gamma" : uniform(0,2),
               "model_use__max_depth": randint(1,15),
               "model_use__colsample_bytree": randint(0.1,1),
               "model_use__subsample": [0.2, 0.4, 0.5],
               "model_use__reg_alpha": uniform(0,1),
               "model_use__reg_lambda": uniform(1,10),
               "model_use__min_child_weight": randint(1,10),
               "model_use__n_estimators": randint(1000,3000)

                    },  #param depending of the model to use
                cv=35,
                scoring='f1',
                n_iter = 10,
                n_jobs = -1 )


            grid_search.fit(self.X_train, self.y_train)

            self.pipeline = grid_search.best_estimator_
            self.grid_params = grid_search.get_params

            self.set_tag('model_used', self.pipeline)



    @simple_time_tracker
    def train(self):
        tic = time.time()
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)
        # mlflow logs
        self.mlflow_log_metric("train_time", int(time.time() - tic))
        self.set_tag('tag_instance', self.tag)

    def evaluate(self):
        f1_train = self.compute_f1(self.X_train, self.y_train)
        precision_train = self.compute_precision(self.X_train, self.y_train)

        self.mlflow_log_metric("f1score_train", f1_train)
        self.mlflow_log_metric("precision_train", precision_train)

        if self.split:
            f1_val = self.compute_f1(self.X_val, self.y_val, show=True)
            precision_val = self.compute_precision(self.X_val, self.y_val, show=True)
            self.mlflow_log_metric("f1score_val", f1_val)
            self.mlflow_log_metric("precision_val", precision_val)
            print(colored("f1 train: {} || f1 val: {}".format(f1_train, f1_val), "yellow"))
            print(colored("precision train: {} || precision val: {}".format(precision_train, precision_val), "yellow"))
        else:
            print(colored("f1 train: {}".format(f1_train), "blue"))
            print(colored("precision train: {}".format(precision_train), "blue"))

    def compute_f1(self, X_test, y_test, show=False):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        if show:
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(self.y_val.shape[0]), "blue")) #Aumentar tamaño de muestra de validacion
        f1 = compute_f1(y_pred, y_test)
        return round(f1, 3)


    def compute_precision(self, X_test, y_test, show=False):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        if show:
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(self.y_val.shape[0]), "blue")) #Aumentar tamaño de muestra de validacion
        precision = compute_precision(y_pred, y_test)
        return round(precision, 3)


    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'Final_model.joblib')
        print(colored("final.joblib saved locally", "green"))

    ### MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def log_estimator_params(self):
        reg = self.get_estimator()
        self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        params = reg.get_params()
        for k, v in params.items():
            self.mlflow_log_param(k, v)

    def set_tag(self, key, value):
        if self.mlflow:
            self.mlflow_client.set_tag(self.mlflow_run.info.run_id, key, value)

    def log_kwargs_params(self):
        if self.mlflow:
            for k, v in self.kwargs.items():
                self.mlflow_log_param(k, v)

    def log_machine_specs(self):
        cpus = multiprocessing.cpu_count()
        mem = virtual_memory()
        ram = int(mem.total / 1000000000)
        self.mlflow_log_param("ram", ram)
        self.mlflow_log_param("cpus", cpus)





if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Get and clean data
    experiment = "Invesscience_batch_#463"


    #Change the reference HERE !!!


    reference = 'a'
    year= '2009'
    target_to_drop = 'exit'

    df = get_data_filled(reference=reference,target_to_drop =target_to_drop , year = year)
    y_train = df["target"]
    X_train = df.drop(columns =['target']) #Change when we have categorical var




    for i in range(1):
        for estimator_iter in ['voting'
                              #  'SGDC'
                               #'xgboost',
                                #'GradientBoostingClassifier',
                                #'LogisticRegression'
                                #'SVC',
                                 #'adaboost',
                                 #'DecisionTree'
                                # 'RandomForestClassifier'
                                 ]:

    #ADABOOST : DecisionTree()

            params = dict(tag_description=f'[MODEL FINAL]{estimator_iter}][{year}][{reference}]', reference =reference, year = year ,estimator = estimator_iter,
                estimator_params ={ 'weights' :[6, 2, 5, 3, 4]},
                local=False, split=True,  mlflow = True, experiment_name=experiment,
                imputer= 'SimpleImputer', imputer_params = {'strategy': 'most_frequent'},
                  grid_search_choice= False, smote=True) #agregar


    #'n_neighbors':21, 'weights': 'distance'

            print("############   Loading Data   ############")


            #df= df[df.country_code=='USA']



#[3, 1, 3, 2, 3]
#[6, 2, 5, 3, 4]
#[7, 4, 5, 3, 5]
            #del df
            print("shape: {}".format(X_train.shape))
            print("size: {} Mb".format(X_train.memory_usage().sum() / 1e6))
            # Train and save model, locally and
            t = Trainer(X=X_train, y=y_train, **params)
            #del X_train, y_train


            print(colored("############  Training model   ############", "red"))
            t.train()
            print(colored("############  Evaluating model ############", "blue"))
            t.evaluate()
            print(colored("############   Saving model    ############", "green"))
            t.save_model()


 ################------------Params founded ------#####################################################################

            #High precision more variability

            #Best DecisionTree
            # DecisionTreeClassifier(class_weight='balanced', max_depth=3.853659650929652, max_features='log2',
             #min_samples_split=0.2130615824774026, min_weight_fraction_leaf=0.40855752460926786

             #Best SGDC ()
             #SGDClassifier
             #estimator_params ={'loss': 'epsilon_insensitive' ,'class_weight': 'balanced',alpha=0.25302306090332494,
             #class_weight='balanced', early_stopping=True, epsilon=0.7399967340475004, eta0=0.0001,

             #learning_rate='constant', loss='huber', n_iter_no_change=10, validation_fraction=0.3

            #xgboost
            #xgboost(learning_rate=0.478977150664321, max_delta_step=0, max_depth=5, min_child_weight=9, missing=nan, monotone_constraints='()', n_estimators=119, n_jobs=12, nthread=12, num_parallel_tree=1, random_state=22, reg_alpha=0, reg_lambda=1, scale_pos_weight=4, seed=22,
            #subsample=0.5439148763175726, tree_method='exact, verbosity=None)


#learning_rate=0.478977150664321, max_depth=5, min_child_weight=9,  n_estimators=119, nthread=12, num_parallel_tree=1, random_state=22, scale_pos_weight=4, seed=22,
 #           subsample=0.5439148763175726, tree_method='exact'
