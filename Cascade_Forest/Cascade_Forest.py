from Cascade_Forest.Layer import layer as Layer
from sklearn.externals import joblib
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import os

warnings.filterwarnings("ignore")


def predict_prob_worker1(clf, x):
    return clf.predict_proba(x)


def predict_prob_worker2(clf, x_growing, y_growing, x_estimating):
    clf.fit(x_growing, y_growing)
    return clf.predict_proba(x_estimating)


class cascade_forest:
    def __init__(
        self,
        cv=5,
        tolerance=0,
        stratify=True,
        random_state=None,
        n_jobs=1,
        metrics="accuracy",
        num_layers=10,
        test_size=0.2,
        n_spilts=3,
    ):
        """

        :param cv: the number of k-fold
        :param tolerance: how much improvement of metrics will lead the cascade structure grow deeper,default:0.
        :param stratify: for train_test_spilt and k-fold cross validation,default:True.
        :param random_state: or random state,default:None.
        :param n_jobs: for parallel computation,default:1,-1 for parallel computation.
        :param metrics: the metrics for validation step,decide how does the cascade structure grow,default:accuracy.
        :param num_layers:
        :param test_size: how much of the train_data will be used as test data in validation step,default:0.2.
        :param n_spilts: for k-fold cross validation,default:3.
        """
        self.num_layers = num_layers
        self.metrics = metrics
        self.n_jobs = n_jobs
        self.stratify = stratify
        "record the optimal number of levels"
        self.num_levels = 0
        self.cv = cv
        self.tolerance = tolerance
        self.random_state = random_state
        self.le = preprocessing.LabelEncoder()
        self.estimator_set2 = []
        self.x_train_set = []
        self.x_test_set = []
        self.y_train = []
        self.record = 0
        self.pred_output = []
        self.test_size = test_size
        self.n_splits = n_spilts
        self.x_growing_set = []
        self.y_growing_set = []
        self.x_estimating_set = []
        self.y_estimating_set = []
        self.output = []
        self.trained_classifier = []
        self.Accu = []
        " record the grade"
        self.grade = 0

    def add(self, layer: Layer):
        """
        :param layer: should be the instance of Layer
        :return: the set of untrained estimators
        'add' function is used for initializing the cascade structure
        """
        if not isinstance(layer, Layer):
            raise ValueError("{} must be an object of {}".format(layer, Layer))
        self.estimator_set2.append(layer)

    def create_estimator_set(self):
        """
        :return:
        add prefix for each estimators,in order to differentiating the different layer.
        """
        for id in range(1, len(self.estimator_set2) + 1):
            self.estimator_set2[id - 1] = self.estimator_set2[id - 1].add_prefix(
                **self.estimator_set2[id - 1].config
            )

    def grow_estimate_split(self, x_train, y_train):
        """
        :param x_train: train data
        :param y_train: label
        :return: splitted data

        spilt the train data into estimating data and growing data in order to do the validation step,
        for multiclass classification problem,we usually set the 'Stratify' to True.
        """
        if self.stratify is True:
            x_growing, x_estimating, y_growing, y_estimating = train_test_split(
                x_train,
                y_train,
                test_size=self.test_size,
                shuffle=True,
                stratify=y_train,
                random_state=self.random_state,
            )
        else:
            x_growing, x_estimating, y_growing, y_estimating = train_test_split(
                x_train,
                y_train,
                test_size=self.test_size,
                shuffle=True,
                stratify=None,
                random_state=self.random_state,
            )
        # stratify = y_train
        assert len(np.unique(y_growing)) == len(
            np.unique(y_estimating)
        ), "Number of classes is not equal in growing and estimating dataset,try to set stratify to true "
        return x_growing, x_estimating, y_growing, y_estimating

    def get_augmented_array(self, clf, x, y):
        """
        :param clf: classifier
        :param x: train data
        :param y: label
        :return: class vector,trained classifier
        get the class vector using k-fold cross validation
        """

        if self.stratify is True:
            kf = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
        else:
            kf = KFold(
                n_splits=self.n_splits, random_state=self.random_state, shuffle=True
            )

        x_dim = x.shape[0]
        y_dim = len(np.unique(y))

        augmented_array = np.zeros((x_dim, y_dim), dtype="float16")

        for k, (train_index, test_index) in enumerate(kf.split(x, y)):
            assert len(np.unique(y)) == len(
                np.unique(y[train_index])
            ), "Number of classes is not equal in the phase of cross validation,try to set stratify to true"
            clf.fit(x[train_index], y[train_index])

            pred_prob = clf.predict_proba(x[test_index])

            augmented_array[test_index, :] += pred_prob

        clf.fit(x, y)
        return augmented_array, clf

    def next_input(
        self,
        x,
        y=None,
        classifier_set=None,
        phase="validate",
        estimator_set=None,
        grade=0,
        level_id=0,
    ):

        """
       :param x: train data
       :param y: label
       :param classifier_set: trained classifier set.
       :param phase: determine in validation step or train step.
       :param estimator_set: untrained classifier set.
       :param grade: the grade of current level
       :param level_id: index of the level
       :return: the input for the next layer
        """
        trained_classifier = []
        if phase == "validate":
            "in validation step"
            if y is not None:
                "training on growing set"
                next_input = []
                if self.n_jobs == 1:
                    for name, clf in estimator_set.items():

                        if clf.random_state is None:
                            clf.random_state = np.random.random_integers(1, 1000) + int(
                                abs(
                                    hash("{}{}{}".format(name, level_id, grade))
                                    / 10 ** 15
                                )
                            )

                        "is used for getting the random state number"
                        setattr(
                            self,
                            "{}{}{}".format(name, level_id, grade),
                            clf.random_state,
                        )
                        "record the random state of each classifier at each layer,in order to reuse it in train step"

                        augmented_array, clf = self.get_augmented_array(clf, x, y)

                        next_input.append(augmented_array)

                        trained_classifier.append(clf)
                    next_input.append(self.x_growing_set[grade])
                else:
                    "parallel computation"
                    modified_clf = {}
                    for name, clf in estimator_set.items():

                        if clf.random_state is None:
                            clf.random_state = np.random.random_integers(1, 1000) + int(
                                abs(
                                    hash("{}{}{}".format(name, level_id, grade))
                                    / 10 ** 15
                                )
                            )

                        setattr(
                            self,
                            "{}{}{}".format(name, level_id, grade),
                            clf.random_state,
                        )

                        modified_clf[name] = clf

                    results = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
                        delayed(self.get_augmented_array)(clf, x, y)
                        for clf in modified_clf.values()
                    )

                    for each in results:
                        augmented_array = each[0]
                        clf = each[1]
                        next_input.append(augmented_array)
                        trained_classifier.append(clf)

                    next_input.append(self.x_growing_set[grade])

            else:
                "estimating on estimating set"
                next_input = []
                if self.n_jobs == 1:
                    for clf in classifier_set:
                        next_input.append(clf.predict_proba(x))
                    next_input.append(self.x_estimating_set[grade])
                else:
                    results = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
                        delayed(predict_prob_worker1)(clf, x) for clf in classifier_set
                    )
                    next_input = results

                    next_input.append(self.x_estimating_set[grade])

        else:
            "in training step"
            if y is not None:
                "training on training set"
                next_input = []
                if self.n_jobs == 1:

                    for name, clf in estimator_set.items():

                        clf.random_state = getattr(
                            self, "{}{}{}".format(name, level_id, grade)
                        )
                        "use the same random state number as in validation step"

                        augmented_array, clf = self.get_augmented_array(clf, x, y)

                        next_input.append(augmented_array)

                        trained_classifier.append(clf)
                    next_input.append(self.x_train_set[grade])
                else:
                    modified_clf = {}
                    for name, clf in estimator_set.items():
                        clf.random_state = getattr(
                            self, "{}{}{}".format(name, level_id, grade)
                        )
                        modified_clf[name] = clf

                    results = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
                        delayed(self.get_augmented_array)(clf, x, y)
                        for clf in modified_clf.values()
                    )

                    for each in results:
                        augmented_array = each[0]
                        clf = each[1]
                        next_input.append(augmented_array)
                        trained_classifier.append(clf)

                    next_input.append(self.x_train_set[grade])

            else:
                "estimating on testing set"
                next_input = []
                if self.n_jobs == 1:
                    for clf in classifier_set:
                        next_input.append(clf.predict_proba(x))
                    next_input.append(self.x_test_set[grade])
                else:
                    results = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
                        delayed(predict_prob_worker1)(clf, x) for clf in classifier_set
                    )
                    next_input = results
                    next_input.append(self.x_test_set[grade])

        next_input = np.hstack(next_input)
        "stack the class vector with the original feature along horizontal axis"

        self.trained_classifier = trained_classifier

        return next_input

    def validate_accu(
        self,
        estimator_set,
        current_x_growing,
        current_y_growing,
        current_x_estimating,
        y_estimating,
    ):
        """
       :param estimator_set: untrained classifier
       :param current_x_growing: x_growing
       :param current_y_growing: y_growing
       :param current_x_estimating: x_estimating
       :param y_estimating: y_estimating
       :return: score
       get the score of validation step
       """
        pred_prob = []
        if self.n_jobs == 1:
            for name, clf in estimator_set.items():
                clf.fit(current_x_growing, current_y_growing)
                pred_prob.append(clf.predict_proba(current_x_estimating))
        else:
            results = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
                delayed(predict_prob_worker2)(
                    clf, current_x_growing, current_y_growing, current_x_estimating
                )
                for name, clf in estimator_set.items()
            )
            pred_prob = results

        pred_prob = np.mean(pred_prob, axis=0)
        pred = np.argmax(pred_prob, axis=1)
        if self.metrics == "accuracy":
            score = accuracy_score(y_true=y_estimating, y_pred=pred)
        if self.metrics == "MCC":
            from sklearn.metrics import matthews_corrcoef

            score = matthews_corrcoef(y_true=y_estimating, y_pred=pred)

        return score

    def fit(self, x_train, y_train):
        """
       :param x_train: X_train,should looks like a list of numpy array,[arr1] or [arr1,arr2,arr3]
       :param y_train: y_train
       :return: trained cascade structure
       """

        self.create_estimator_set()
        "initialize the configuration"
        self.x_train_set = x_train
        self.y_train = y_train

        # assert isinstance(x_train, np.ndarray), 'x_train is not numpy array!'
        # assert isinstance(y_train, np.ndarray), 'y_train is not numpy array!'

        grade = 0
        "initialize the grade "
        for each_x_profile in self.x_train_set:
            current_x_growing, current_x_estimating, current_y_growing, current_y_estimating = self.grow_estimate_split(
                each_x_profile, self.y_train
            )
            "split the training set into growing set and estimating set"
            self.x_growing_set.append(current_x_growing)
            self.x_estimating_set.append(current_x_estimating)
            self.y_growing_set.append(current_y_growing)
            self.y_estimating_set.append(current_y_estimating)

        print("[current_x_growing].shape = {} ".format(self.x_growing_set[grade].shape))
        print(
            "[current_x_estimating].shape = {} ".format(
                self.x_estimating_set[grade].shape
            )
        )

        current_x_growing = self.x_growing_set[grade]
        current_y_growing = self.y_growing_set[grade]
        current_x_estimating = self.x_estimating_set[grade]
        current_y_estimating = self.y_estimating_set[grade]

        best_accu = self.validate_accu(
            self.estimator_set2[grade],
            current_x_growing,
            current_y_growing,
            current_x_estimating,
            current_y_estimating,
        )
        print(
            "{}th level is done,current validation score is {:.2f} %".format(
                self.record, best_accu * 100
            )
        )
        "compute the score of the 0th level,which will be used as the benchmark"

        "1 level 1 grade"

        current_x_growing = self.next_input(
            current_x_growing,
            current_y_growing,
            phase="validate",
            estimator_set=self.estimator_set2[self.record],
            grade=grade,
            level_id=self.record + 1,
        )
        print("[current_x_growing].shape = {} ".format(current_x_growing.shape))
        current_x_estimating = self.next_input(
            current_x_estimating,
            classifier_set=self.trained_classifier,
            phase="validate",
            grade=grade,
        )
        print("[current_x_estimating].shape = {} ".format(current_x_estimating.shape))

        current_accu = self.validate_accu(
            self.estimator_set2[self.record],
            current_x_growing,
            current_y_growing,
            current_x_estimating,
            current_y_estimating,
        )

        self.record += 1
        print(
            "{}th level {} grade is done,current validation score is {:.2f} %".format(
                self.record, grade + 1, current_accu * 100
            )
        )

        while (
            current_accu - best_accu >= self.tolerance and self.record < self.num_layers
        ):

            self.grade = grade
            self.num_levels = self.record
            best_accu = current_accu

            if grade < len(self.x_train_set) - 1:
                grade += 1
            else:
                grade = 0
                self.record += 1
                if self.record == self.num_layers:
                    break
                else:
                    pass

            current_x_growing = self.next_input(
                current_x_growing,
                current_y_growing,
                phase="validate",
                estimator_set=self.estimator_set2[self.record],
                grade=grade,
                level_id=self.record,
            )

            print("[current_x_growing].shape = {} ".format(current_x_growing.shape))

            current_x_estimating = self.next_input(
                current_x_estimating,
                classifier_set=self.trained_classifier,
                phase="validate",
                grade=grade,
            )
            print(
                "[current_x_estimating].shape = {} ".format(current_x_estimating.shape)
            )

            current_y_growing = self.y_growing_set[grade]
            current_y_estimating = self.y_estimating_set[grade]

            current_accu = self.validate_accu(
                self.estimator_set2[self.record],
                current_x_growing,
                current_y_growing,
                current_x_estimating,
                current_y_estimating,
            )
            print(
                "{}th level {} grade is done,current validation score is {:.2f} %".format(
                    self.record, grade + 1, current_accu * 100
                )
            )

        print(
            "At {}th {} grade level,we met early stopping,the best validation score is {:.2f} %".format(
                self.num_levels, self.grade + 1, best_accu * 100
            )
        )
        # self.grade = grade

        x_train = self.x_train_set[0]
        for level_id in range(0, self.record + 1):
            "after validation step,we get the optimal number of level and number of grade,and then we retrain the model on entire X_train."
            # if level_id != self.record:
            if level_id == 0:
                grade_id = 0
                for name, clf in self.estimator_set2[level_id].items():
                    clf.fit(x_train, y_train)
                    joblib.dump(
                        clf,
                        os.getcwd() + "/model/{}_{}th_level.pkl".format(name, level_id),
                    )
                x_train = self.next_input(
                    x_train,
                    y_train,
                    phase="train",
                    estimator_set=self.estimator_set2[level_id],
                    grade=grade_id,
                    level_id=level_id + 1,
                )
            else:
                if level_id != self.record:
                    for grade_id in range(0, len(self.x_train_set)):
                        for name, clf in self.estimator_set2[level_id].items():
                            clf.fit(x_train, y_train)
                            joblib.dump(
                                clf,
                                os.getcwd()
                                + "/model/{}_{}th_level_{}_grade.pkl".format(
                                    name, level_id, grade_id + 1
                                ),
                            )
                        x_train = self.next_input(
                            x_train,
                            y_train,
                            phase="train",
                            estimator_set=self.estimator_set2[level_id],
                            grade=grade_id,
                            level_id=level_id,
                        )
                else:
                    for grade_id in range(0, grade):
                        for name, clf in self.estimator_set2[level_id].items():
                            clf.fit(x_train, y_train)
                            joblib.dump(
                                clf,
                                os.getcwd()
                                + "/model/{}_{}th_level_{}_grade.pkl".format(
                                    name, level_id, grade_id + 1
                                ),
                            )
                        x_train = self.next_input(
                            x_train,
                            y_train,
                            phase="train",
                            estimator_set=self.estimator_set2[level_id],
                            grade=grade_id,
                            level_id=level_id,
                        )

    def predict(self, x_test):

        self.x_test_set = x_test
        all_classifier = []

        # print(self.x_test_set.shape)

        for level_id in range(0, self.num_levels + 1):

            if level_id == 0:
                temp2 = []
                for name, clf in self.estimator_set2[level_id].items():
                    temp2.append(
                        joblib.load(
                            os.getcwd()
                            + "/model/{}_{}th_level.pkl".format(name, level_id)
                        )
                    )

                all_classifier.append(temp2)

            else:
                if level_id != self.num_levels:
                    # print("lallalal")
                    for grade_id in range(0, len(self.x_train_set)):
                        temp3 = []
                        for name, clf in self.estimator_set2[level_id].items():
                            temp3.append(
                                joblib.load(
                                    os.getcwd()
                                    + "/model/{}_{}th_level_{}_grade.pkl".format(
                                        name, level_id, grade_id + 1
                                    )
                                )
                            )

                        all_classifier.append(temp3)
                else:
                    for grade_id in range(0, self.grade + 1):
                        temp4 = []
                        for name, clf in self.estimator_set2[level_id].items():
                            temp4.append(
                                joblib.load(
                                    os.getcwd()
                                    + "/model/{}_{}th_level_{}_grade.pkl".format(
                                        name, level_id, grade_id + 1
                                    )
                                )
                            )
                        all_classifier.append(temp4)

        def get_output(current_input, classifier_set):

            output_lst = []
            if self.n_jobs == 1:
                for clf in classifier_set:
                    output_lst.append(clf.predict_proba(current_input))

            else:
                from multiprocessing.pool import Pool

                args = []
                for clf in classifier_set:
                    args.append((clf, current_input))
                with Pool(16) as pool:
                    output_lst = pool.starmap(predict_prob_worker1, args)

            output = sum(output_lst) / len(output_lst)
            return output

        output_lst = []
        current_input = x_test[0]
        clf_id = 0
        for level_id in range(0, self.num_levels + 1):
            if level_id == 0:
                next_input_data = self.next_input(
                    current_input,
                    classifier_set=all_classifier[clf_id],
                    phase="test",
                    grade=0,
                )

                output_lst = get_output(current_input, all_classifier[clf_id])
                current_input = next_input_data
            else:
                if level_id != self.num_levels:
                    for grade_id in range(0, len(x_test)):
                        clf_id += 1
                        next_input_data = self.next_input(
                            current_input,
                            classifier_set=all_classifier[clf_id],
                            phase="test",
                            grade=grade_id,
                        )
                        # print(current_input.shape)
                        output_lst = get_output(current_input, all_classifier[clf_id])
                        current_input = next_input_data

                else:
                    for grade_id in range(0, self.grade + 1):
                        clf_id += 1
                        next_input_data = self.next_input(
                            current_input,
                            classifier_set=all_classifier[clf_id],
                            phase="test",
                            grade=grade_id,
                        )
                        output_lst = get_output(current_input, all_classifier[clf_id])
                        current_input = next_input_data

        self.output = output_lst

    def score(self, y_validate):

        pred = np.argmax(self.output, axis=1)

        if self.metrics == "accuracy":
            score = accuracy_score(y_true=y_validate, y_pred=pred)
        if self.metrics == "MCC":
            from sklearn.metrics import matthews_corrcoef

            score = matthews_corrcoef(y_true=y_validate, y_pred=pred)

        return score
