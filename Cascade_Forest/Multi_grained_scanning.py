import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


class scanner:
    def __init__(
        self,
        clf_set,
        n_splits=3,
        stratify=True,
        random_state=None,
        walk_length=(1, 1),
        num_walks=1,
        scale=(1, 1),
        window_size=(1, 1),
        p=1,
        q=1,
    ):
        self.window_size = window_size
        self.clf_set = clf_set
        self.n_splits = n_splits
        self.stratify = stratify
        self.random_state = random_state
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.scale = scale

    def slicing(self, x_train, x_test, each_window_size):

        n_features = x_train.shape[1]
        subset_train = []
        subset_test = []
        column_index = []
        #window_path = []
        if type(each_window_size) is int:
            assert len(x_train.shape) == 2, "the shape of x_train must be 2d!"
            assert len(x_test.shape) == 2, "the shape of x_test must be 2d!"
            if n_features < each_window_size:
                msg = "`window size` must be in range 1 to {} but got value {}"
                raise ValueError(msg.format(n_features, each_window_size))
            else:
                start_index = [i for i in range(0, n_features - each_window_size + 1)]
                end_index = [i for i in range(each_window_size - 1, n_features)]

            for i, j in zip(start_index, end_index):
                # subset_train.append(x_train[:, i : j + 1])
                # subset_test.append(x_test[:, i : j + 1])
                #window_path.append(list(range(i, j+1)))
                column_index.append(list(np.arange(i, j+1)))

        else:
            assert len(x_train.shape) == 3, "the shape of x_train must be 3d!"
            assert len(x_test.shape) == 3, "the shape of x_test must be 3d!"
            if x_train.shape[0] < each_window_size[0]:
                msg = "`window's length ` must be in range 1 to {} but got value {}"
                raise ValueError(msg.format(x_train.shape[0], each_window_size[0]))
            elif x_train.shape[1] < each_window_size[1]:
                msg = "`window's width` must be in range 1 to {} but got value {}"
                raise ValueError(msg.format(x_train.shape[1], each_window_size[1]))

            fig = x_train[0]
            row_max = fig.shape[0]
            column_max = fig.shape[1]
            row_index = [i for i in range(0, row_max - each_window_size[0] + 1)]
            col_index = [i for i in range(0, column_max - each_window_size[1] + 1)]

            for row_id in row_index:
                for col_id in col_index:
                    pool_train = []
                    for fig_train in x_train:
                        pool_train.append(
                            fig_train[
                                row_id : row_id + each_window_size[0],
                                col_id : col_id + each_window_size[1],
                            ].ravel()
                        )
                    subset_train.append(pool_train)

            for row_id in row_index:
                for col_id in col_index:
                    pool_test = []
                    for fig_test in x_test:
                        pool_test.append(
                            fig_test[
                                row_id : row_id + each_window_size[0],
                                col_id : col_id + each_window_size[1],
                            ].ravel()
                        )
                    subset_test.append(pool_test)

        #return subset_train, subset_test,window_path
        return column_index

    def get_class_vector(self, clf, x, y):

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

        class_vector_array = np.zeros((x_dim, y_dim), dtype="float32")

        for k, (train_index, test_index) in enumerate(kf.split(x, y)):
            assert y_dim == len(
                np.unique(y[train_index])
            ), "Number of classes is not equal in the phase of cross validation,try to set stratify to true"

            clf.fit(x[train_index], y[train_index])

            pred_prob = clf.predict_proba(x[test_index])

            class_vector_array[test_index, :] += pred_prob

        clf.fit(x, y)

        return class_vector_array, clf

    def transform_feature(self, x_train, y_train, x_test):

        transformed_train = []
        transformed_test = []

        # window_path = []
        for each_window_size in self.window_size:
            #subset_train, subset_test, window_temp_path = self.slicing(x_train, x_test, each_window_size)
            column_index = self.slicing(x_train, x_test, each_window_size)
            class_vector_train = []
            class_vector_test = []

            #window_path.append(window_temp_path)

            for clf in self.clf_set:

                for i in range(0, len(column_index)):

                    each = column_index[i]
                    column_lst = [int(i) for i in each]
                    each_subset_train = x_train[:, column_lst]
                    each_subset_test = x_test[:, column_lst]

                # for each_subset_train, each_subset_test in zip(
                #     subset_train, subset_test
                # ):

                    class_vector_array, clf = self.get_class_vector(
                        clf, np.array(each_subset_train), np.array(y_train)
                    )
                    class_vector_train.append(class_vector_array)
                    class_vector_test.append(
                        clf.predict_proba(np.array(each_subset_test))
                    )

            transformed_train.append(np.hstack(class_vector_train))
            transformed_test.append(np.hstack(class_vector_test))

        return transformed_train, transformed_test

    def get_subset(self, G, walk_length, num_walks, scale):

        from node2path.Node2Path import Node2Path

        model = Node2Path(
            G,
            walk_length=walk_length,
            num_walks=num_walks,
            p=self.p,
            q=self.q,
            workers=2,
        )

        column_index = model.get_path()[:scale]

        return column_index

    def graph_embedding(self, G, x_train, y_train, x_test):

        transformed_train = []
        transformed_test = []
        path = []
        for each_walk_length, each_scale in zip(self.walk_length, self.scale):

            column_index = self.get_subset(
                G, each_walk_length, self.num_walks, each_scale
            )

            path.append(column_index)

            class_vector_train = []
            class_vector_test = []

            for clf in self.clf_set:

                for i in range(0, len(column_index)):

                    each = column_index[i]
                    column_lst = [int(i) for i in each]
                    each_subset_train = x_train[:, column_lst]
                    each_subset_test = x_test[:, column_lst]
                    class_vector_array, clf = self.get_class_vector(
                        clf, np.array(each_subset_train), np.array(y_train)
                    )
                    class_vector_train.append(class_vector_array)
                    class_vector_test.append(
                        clf.predict_proba(np.array(each_subset_test))
                    )

            transformed_train.append(np.hstack(class_vector_train))
            transformed_test.append(np.hstack(class_vector_test))

        return transformed_train, transformed_test
