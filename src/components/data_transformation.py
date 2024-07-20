# #handle missing values
# #handle outliers
# #encode all categorical columns into numerical
# #Handle imbalanced dataset

# from dataclasses import dataclass
# import os
# import sys
# import pandas as pd
# import numpy as np
# from src.logger import logging
# from src.exception import CustomException
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from src.utils import save_object

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")

# class get_data_transformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()

#     def get_data_transformation_object(self):
#         try:
#             logging.info("Data transformation initiated")

#             # Define numerical and categorical columns
#             numerical_columns = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']
#             categorical_columns = ['workclass', 'education', 'marital_status', 'occupation',
#                                    'relationship', 'race', 'sex', 'native.country']

#             # Numerical pipeline
#             num_pipeline = Pipeline(
#                 steps=[
#                     ("imputer", SimpleImputer(strategy="median")),
#                     ("scaler", StandardScaler())
#                 ]
#             )

#             # Categorical pipeline
#             cat_pipeline = Pipeline(
#                 steps=[
#                     ("imputer", SimpleImputer(strategy="most_frequent")),
#                     ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
#                 ]
#             )

#             # Combine pipelines into a preprocessor
#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     ("num", num_pipeline, 
# ),
#                     ("cat", cat_pipeline, categorical_columns)
#                 ]
#             )

#             return preprocessor
#         except Exception as e:
#             raise CustomException(e, sys)

#     def remove_outliers(self, columns, df):
#         try:
#             if columns not in df.columns:
#                 raise KeyError(f"Column '{columns}' not found in DataFrame")

#             Q1 = df[columns].quantile(0.25)
#             Q3 = df[columns].quantile(0.75)

#             iqr = Q3 - Q1
#             upper_limit = Q3 + 1.5 * iqr
#             lower_limit = Q1 - 1.5 * iqr

#             df.loc[df[columns] > upper_limit, columns] = upper_limit
#             df.loc[df[columns] < lower_limit, columns] = lower_limit

#             return df
#         except KeyError as e:
#             logging.error(f"KeyError in remove_outliers: {e}", exc_info=True)
#             raise CustomException(e, sys)
#         except Exception as e:
#             logging.error(f"Error in remove_outliers method: {e}", exc_info=True)
#             raise CustomException(e, sys)

#     def initialize_data_transformation(self, train_path, test_path):
#         try:
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)

#             numerical_columns = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']

#             for column in numerical_columns:
#                 train_df = self.remove_outliers(column, train_df)
#                 test_df = self.remove_outliers(column, test_df)

#             preprocessor_obj = self.get_data_transformation_object()

#             target_column_name = "income"
#             drop_columns = [target_column_name]

#             logging.info("Splitting training data into features and target")
#             input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
#             target_feature_train_df = train_df[target_column_name]

#             logging.info("Splitting testing data into features and target")
#             input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
#             target_feature_test_df = test_df[target_column_name]

#             # Apply transformations
#             input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

#             # Combine features and target
#             train_array = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
#             test_array = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

#             save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor_obj)

#             return train_array, test_array, self.data_transformation_config.preprocessor_obj_file_path
#         except Exception as e:
#             logging.error(f"Error in initialize_data_transformation method: {e}", exc_info=True)
#             raise CustomException(e, sys)

# from dataclasses import dataclass
# import os 
# import sys
# import pandas as pd 
# import numpy as np 
# from src.logger import logging 
# from src.exception import CustomException 
# from sklearn.preprocessing import StandardScaler,OneHotEncoder
# from sklearn.impute import SimpleImputer 
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from src.utils import save_object


# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join("artifacts/data_transformation","preprocessor.pkl")
    

# class get_data_transformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()
        
#     def get_data_transformation_object(self):
#         try:
            
#             logging.info("Data transformation initiated")
#             numerical_columns = ['age', 'workclass', 'education_num', 'marital_status', 'occupation',
#             'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
#             'hours_per_week', 'native_country']
            
#             num_pipeline = Pipeline(
#                 steps = [
#                     ("imputer",SimpleImputer(strategy="median")),
#                     ("scaler",StandardScaler())
#                 ]
#             )
            
#             cat_pipeline = Pipeline(
#                 steps = [
#                     ("imputer",SimpleImputer(strategy="most_frequent")),
#                     ("OneHotEncoder", OneHotEncoder(handle_unknown='ignore'))
#                     #("one_hot_encoder",SimpleImputer(strategy="most_frequent")),
#                 ]
#             )
            
#             preprocessor = ColumnTransformer([ 
#                 ("num_pipeline",num_pipeline,numerical_columns,"cat_pipeline",cat_pipeline)
#                 ])
            
#             return preprocessor
#         except Exception as e:
#             raise CustomException(e,sys)
        
#     def remove_outliers(self,columns,df):
#         try:
#             Q1 = df[columns].quantile(0.25)
#             Q3 = df[columns].quantile(0.75)
            
#             iqr = Q3-Q1 
            
#             upper_limit = Q3 + 1.5 * iqr
#             lower_limit = Q1 - 1.5 * iqr 
            
#             df.loc[(df[columns]>upper_limit),columns] = upper_limit
#             df.loc[(df[columns]<lower_limit),columns] = lower_limit
            
#             return df 
        
#         except Exception as e:
#             logging.info("Outliers handling code")
#             raise CustomException(e,sys)
        
#     def initialize_data_transformation(self,train_path,test_path):
#         try:
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)
            
#             numerical_columns = ['age', 'workclass', 'education_num', 'marital_status', 'occupation',
#             'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
#             'hours_per_week', 'native_country']
            
#             for columns in numerical_columns:
#                 self.remove_outliers(columns=columns,df=train_df)
                
#                 logging.info("Outliers removed")
            
#             preprocessor_obj = self.get_data_transformation_object()
            
#             target_column_name = "income"
#             drop_columns = [target_column_name]
            
#             logging.info("Splitting training data into dependent and target feature")
#             input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
#             target_feature_train_df = train_df[target_column_name]
            
#             logging.info("Splitting testing data into dependent and target feature")
#             input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
#             target_feature_test_df = test_df[target_column_name]
            
#             #Applying transformation on train and test data
#             input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr = preprocessor_obj.fit_transform(input_feature_test_df )
            
#             #Applying preprocessor object on train and test data
            
#             train_array = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
#             test_array = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
#             save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,obj = preprocessor_obj)
            
#             return(train_array,test_array,
#             self.data_transformation_config.preprocessor_obj_file_path)
                
                
                

#         except Exception as e:
#                 raise CustomException(e,sys)


from dataclasses import dataclass
import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")

class get_data_transformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data transformation initiated")

            # Define numerical and categorical columns
            numerical_columns = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']
            categorical_columns = ['workclass', 'education_num', 'marital_status', 'occupation',
            'relationship', 'race', 'sex', 'native_country']

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]
            )

            # Combine pipelines into a preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def remove_outliers(self, column, df):
        try:
            if column not in df.columns:
                logging.warning(f"Column '{column}' not found in DataFrame")
                return df

            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)

            iqr = Q3 - Q1
            upper_limit = Q3 + 1.5 * iqr
            lower_limit = Q1 - 1.5 * iqr

            dtype = df[column].dtype
            df[column] = np.where(df[column] > upper_limit, upper_limit.astype(dtype), df[column])
            df[column] = np.where(df[column] < lower_limit, lower_limit.astype(dtype), df[column])

            return df
        except Exception as e:
            logging.error(f"Error in remove_outliers method: {e}", exc_info=True)
            raise CustomException(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            numerical_columns = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']
            categorical_columns = ['workclass', 'education_num', 'marital_status', 'occupation',
                                   'relationship', 'race', 'sex', 'native_country']
            all_columns = numerical_columns + categorical_columns

            missing_columns = [col for col in all_columns if col not in train_df.columns]
            if missing_columns:
                logging.warning(f"Missing columns in DataFrame: {missing_columns}")
                raise CustomException(f"Missing columns in DataFrame: {missing_columns}", sys)

            for column in numerical_columns:
                train_df = self.remove_outliers(column, train_df)
                test_df = self.remove_outliers(column, test_df)

            preprocessor_obj = self.get_data_transformation_object()

            target_column_name = "income"
            drop_columns = [target_column_name]

            logging.info("Splitting training data into features and target")
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            logging.info("Splitting testing data into features and target")
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply transformations
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Combine features and target
            train_array = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor_obj)

            return train_array, test_array, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            logging.error(f"Error in initialize_data_transformation method: {e}", exc_info=True)
            raise CustomException(e, sys)

        
        
        
        
        
        
        
        
        
            
    
