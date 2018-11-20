import pandas as pd
import numpy as np
import pickle

class HeartDiseaseClassifier:

    impute_mean_columns = ['Column1',  'Column4', 'Column5', 'Column8', 'Column10']
    impute_mode_columns = ['Column2', 'Column3', 'Column6', 'Column7', 'Column9']
    impute_regression_columns = ['Column11', 'Column12', 'Column13']

    def __init__(self):

        self._set_model("model/model_decision_tree")
        self._set_normalizer("model/normalizer_dataframe")

        self.imputer_11 = self._load_imputer("model/imputer_11")
        self.imputer_12 = self._load_imputer("model/imputer_12")
        self.imputer_13 = self._load_imputer("model/imputer_13")
        self.imputer_mean = self._load_imputer("model/imputer_mean")
        self.imputer_mode = self._load_imputer("model/imputer_mode")

    def _set_model(self, filename):
        try:
            self.model = pickle.load(open(filename, 'rb'))
        except FileExistsError:
            self.model = None
            raise Exception("File {} doesn't exist !".format(filename))

    def _load_imputer(self, filename): 
        try:
            return pickle.load(open(filename, 'rb'))
        except FileExistsError:
            raise Exception("File {} doesn't exist !".format(filename))

    def _set_normalizer(self, filename):
        try:
            self.normalizer = pickle.load(open(filename, 'rb'))
        except FileExistsError:
            raise Exception("File {} doesn't exist !".format(filename))

    def _impute_missing_value(self, input_features):

        input_features.update(self._impute_using_mean(input_features))
        input_features.update(self._impute_using_mode(input_features))

        input_features = self._impute_column(input_features, 'Column11', self.imputer_11)
        input_features = self._impute_column(input_features, 'Column12', self.imputer_12)
        input_features = self._impute_column(input_features, 'Column13', self.imputer_13)

        input_features = input_features.apply(pd.to_numeric)

        return input_features

    def _impute_using_mean(self, input_features):

        X_impute_mean = input_features[HeartDiseaseClassifier.impute_mean_columns]
        X_impute_mean = self.imputer_mean.transform(X_impute_mean)

        return pd.DataFrame(X_impute_mean, columns=HeartDiseaseClassifier.impute_mean_columns)

    def _impute_using_mode(self, input_features):

        X_impute_mode =  input_features[HeartDiseaseClassifier.impute_mode_columns]
        X_impute_mode = self.imputer_mode.transform(X_impute_mode)

        return pd.DataFrame(X_impute_mode, columns=HeartDiseaseClassifier.impute_mode_columns)

    def _impute_column(self, input_features, column, imputer):

        feature_column = input_features.drop(columns=HeartDiseaseClassifier.impute_regression_columns)
        target_column = input_features[column]

        for i in target_column.index:
            if target_column.loc[i] is np.NAN: 

                input_feature = feature_column.loc[i]
                imputed_value = imputer.predict([input_feature])[0]

                input_features.at[i, column] = imputed_value

        return input_features

    def _normalize(self, input_features):
        from sklearn.preprocessing import normalize

        temp = self.normalizer.copy()
        temp = temp.append(input_features, ignore_index=True)

        input_features_normalized = normalize(temp, axis=0)[-1]
        return pd.DataFrame(input_features_normalized, index=input_features.columns).T

    def predict(self, input_features):

        input_features = pd.DataFrame(input_features, index=["Column"+str(i+1) for i in range(0, len(input_features))]).T
        imputed_feature = self._impute_missing_value(input_features)
        normalized_feature = self._normalize(imputed_feature)

        return self.model.predict(normalized_feature)


if __name__ == "__main__":

    import os

    clf = HeartDiseaseClassifier()
    test_input = [54, 0, 3, 135, 304, 1, 0, 170, 0 ,0, 1, 0, 3]

    output = clf.predict(test_input)
    print("Prediction : {}".format(output))

    try:
        os.system("pause")
    except:
        os.system('read -s -n 1 -p "Press any key to continue..."')