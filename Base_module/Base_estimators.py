from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

class RFC(RandomForestClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class GBC(GradientBoostingClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ETC(ExtraTreesClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
class LR(LogisticRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# # clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
# clf = RF(n_estimators=100, max_depth=2,random_state=0)
# dd = [(3,7),(2,7),(2,7),(2,7),(2,7),(2,7)]
# dd2 = [(4,7),(2,7),(2,7),(2,7),(2,7),(2,7)]
# y = [0,1,1,1,1,1,]
# clf.fit(dd, y)
# rf = RF(n_estimators = 20)
# .predict(dd,y)
# print(clf.predict(dd2))
