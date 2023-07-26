import pandas as pd
#model wrapper
class model_wrapper:
    def __init__(self, models, weights):
        #models is a dictionary of models
        #weights is a dictionary of weights
        self.models = models
        self.weights = weights
    def predict(self, data):
        pred = pd.DataFrame()
        for i in self.models:
            pred[i] = self.models[i].predict(data)
        #weighted average
        pred['weighted'] = 0
        for i in self.weights:
            pred['weighted'] += pred[i] * self.weights[i]
        return pred['weighted']


class feature_wrapper:
    def __init__(self, feature_functions):
        #feature_functions is a dictionary of feature functions
        self.feature_functions = feature_functions
    def transform(self, data):
        features = pd.DataFrame()
        for i in self.feature_functions:
            features[i] = self.feature_functions[i](data)
        return features
    def normaliztion(self, data):
        window = 10000
        mean = data.rolling(window).mean()
        std = data.rolling(window).std()
        return (data - mean) / std
    
#if predict is a method of model, then we can use model.predict(data)
#else we can use model(data)
def gen_result(model, data):
    #use model.predict(data) or model(data) to generate result
    if hasattr(model, 'predict'):
        return model.predict(data)
    else:
        return model(data)