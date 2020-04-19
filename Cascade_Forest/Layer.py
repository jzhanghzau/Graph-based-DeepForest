class layer:

    def __init__(self):
        self.config = {}

    def add(self, **estimators_set):
        self.config = estimators_set

    def add_prefix(self, **estimators_set):

        dic = {"_{}".format(k): v for k, v in estimators_set.items()}

        return dic











