import pandas as pd
import numpy as np
import xgboost as xgb
from data_utils import *

class score_model:
    """Class to serve a data payload to the model. Extracts a patient record, subets a horizon segment from the record and constructs the feature vector for this segment. 
    """
    
    def __init__(self, estimator, horizon= 12, effective_features = None):
        """Initialize the class with the trained estimator, the horizon length the was used in training. The function offers an override of the effective physiological feature.
        
        Parameters
        ----------
        estimator : XGBClassifier object
            The trained estimator to use to score the model.  
        horizon : int
            The length of the prediction horizon. The value should be equal to what was used in model training (default 12).
        effective_features : list
            List of names for the physio features that were used in model training. Default uses a predefined list, otherwise the list of features will overidden with input.
        """
        self.estimator= estimator
        self.horizon = horizon * 60
        if not effective_features:
            self.effective_features= ['measurement_datetime',
                                     'hours_since_birth',
                                     'Monitor Arteriele Bloeddruk Diastole',
                                     'Monitor Arteriele Bloeddruk Mean',
                                     'Monitor Arteriele Bloeddruk Systole',
                                     'Monitor Hartfrequentie',
                                     'Monitor Ademhalingsfrequentie',
                                     'Monitor O2 Saturatie',
                                     'Couveuse Gemeten Temp',
                                     'Monitor Temperatuur 1',
                                     'Monitor Hartfrequentie Pulse',
                                     'Monitor Hartfrequentie Pleth']
        else:
            self.effective_features= effective_features
            
              
    def select_patient(self, df_raw, patient):
        """Selects the full record of the input patient from the dataframe of physio markers. Returns a df with the patient's timeseries record.
        
        Parameters
        ----------
        df_raw : DataFrame
            The dataframe with patient's physio timeseries data.
        patient : int
            The patient's id to extract thier record.
            
        Returns
        -------
        DataFrame
            Df with the input patient's record.
        """
    
        payload = df_raw.loc[df_raw.patient_id == patient]
        payload = payload.set_index('patient_id')
        payload = payload.filter(self.effective_features)

        assert len(payload.measurement_datetime) >= self.horizon , 'patient record must be >= 720 minutes'
    
        return payload

    def extract_horizon(self, payload, mark):
        """Subsets a full patient record into a segment of length = `self.horizon` hours, starting at `mark`. Returns a df with segment of 720 minutes from a patient's record.
        
        Parameters
        ----------
        payload : DataFrame
            Df with one patient's full timseries record.
        mark : int
            The point along the timestamps at which the extracted segment starts.
            
        Returns
        -------
        DataFrame
            The segment of the patient record of length `horizon` to use for prediction.
        """

        payload = payload.set_index('measurement_datetime', append = True)
        payload = payload.sort_index(ascending= True)

        mark *=  60
        slc = pd.IndexSlice
        horizon = payload.loc[slc[ :, mark : mark + self.horizon], :]
        horizon.reset_index(level = 'measurement_datetime', inplace = True)

        assert len(horizon) >= self.horizon, 'Failed to extract 720 minutes, propably bad mark parameter'

        return horizon
    
    @staticmethod
    def get_feat_vec(horizon, obs_df):
        """Constructs features over a patient horizon. Returns a df of one feature vector for one patient.
        
        Calculates the 8 statistical features for the 10 physio markers and adds the patient's gestational age and gender.
        
        Parameters
        ----------
        horizon : DataFrame
            The segmented horizon from a patient's record.
        obs_df : DataFrame
            The observations dataframe to lookup the patients gestational age and gender.
            
        Returns
        -------
        DataFrame
            Df with a single feature vector for one patient. Has the shape (1, 403).
        """
        constructor= construct_features()
        feat_set = constructor.feature_df(horizon, obs_df)
        feat_set = feat_set.sort_index(level=0, axis = 1)
        return feat_set
    

    def get_X(self, horizon, obs_df ):
        """Constucts a 2D array of shape (1, 403) containing the statistical and profile features for a single patient over a single horizon segment. Returns np.array of the patient's feature vector.
                
        Parameters
        ----------
        horizon : DataFrame
            The segmented horizon from a patient's record.
        obs_df : DataFrame
            The observations dataframe to lookup the patients gestational age and gender.
            
        Returns
        -------
        array
            2D array of shape (1, 403) containing the statistical and profile features for a single patient's horizon.
        """
        feat_set= self.get_feat_vec (horizon, obs_df)
        X = feat_set.iloc[0]
        return X
    
    def score(self, X):
        """Produces the model's predicted class for the input `X` array. Returns a string describing the model's predicted class.
        
        Paramaters
        ----------
        X : array
            2D array of shape (1, 403) containing the statistical and profile features for a single patient's horizon.
            
        Returns
        ------- 
        str
            String message describing the model's predicted class.    
        """
        y_pred = self.estimator.predict(X)
        message= 'SEPSIS SUSPICION' if y_pred == 1 else 'CLEAR'
        out = f"----------------\n{message}\n----------------"
        return out
        
    
    def score_strict(self, X, threshold = 0.8):
        """Runs a strict or loose prediction based on an input threshold. Returns a string describing the model's predicted class.
        
        Paramaters
        ----------
        X : array
            2D array of shape (1, 403) containing the statistical and profile features for a single patient's horizon.
        threshold : float
            The prediction probability threshold used to declare a postive class; aka. sepsis event (default 0.80).
            
        Returns
        ------- 
        str
            String message describing the model's predicted class.
        """
        y_pred = self.estimator.predict_proba(X).ravel()
        flags = ['CLEAR', 'SEPSIS SUSPICION']
        message= flags[0] if y_pred[1] < threshold else flags[1]
        out = f"----------------\n{message}\n----------------"
        return out