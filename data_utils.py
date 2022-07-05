import pandas as pd
import numpy as np


class process_raw:
    """Processes the raw df of physiological markers into case and control groups; extracts the horizon segment of a patient's timeseries and constructs the feature vector per patient."""
    
    def __init__(self, hist_length = 24, t_control = 60, horizon = 12, sample = 0, seed = 606):
        """Instantiates a process_raw class instance with a the minimum accepted history record length per patient and the desirable prediction horizon.
        
        Parameters
        ----------
        hist_length : int
            Number of hours defining the minimum length of the data stream as a criterion for patient selection.
        t_control : int
           The control timestamp in hours, randomly selected from the open interval [24:] hrs. Defaults to 60 hours.
        horizon: int
            Defines the prediction horizon. Also the length of patient history prior to t_sepsis to be extracted for feature extraction & model training.
        sample : int
            Defines the size of a subsample of patients. Avoids space & time complexity during prototyping.
            If > 0, a random sample of size = sample is drawn uniformly from the patient_ids set without replacement (defaults to 0; all 2618 patients are processed).
        seed : int
           Defines a global random state for all downstream pseudo operations to maintain reproducibility in data processing (defaults to 606).
            
        """
        assert hist_length >= 0,  "hist_length must be >= 0"
        assert t_control >= 24,  "t_control must be >= 24"
        assert isinstance(sample, int), 'sample type must be an integer'
        
        self.hist_length = hist_length * 60  # convert from hours to minutes.
        self.horizon = horizon * 60
        self.sample = sample
        self.t_control = t_control * 60
        self.seed = seed
        
    @staticmethod
    def record_validity(df_raw, hist_length):
        """Filters patient_ids based on their record length. Patients with records < `hist_length` are discarded.
         
        Parameters
        ---------
        df_raw: DataFrame
            The raw patient timeseries dataframe. Must contain ["patient_id", "measurement_datetime"] columns.
        hist_length : int
            Number of hours defining the minimum length of the data stream for patient selection.
        
        Returns
        -------
        array
            Filtered patient_ids with data streams >= 24 hours in length.
         
         """
        valid_records = df_raw.groupby(by= 'patient_id')['measurement_datetime'].max() >= hist_length
        valid_patient_ids = valid_records.index.values
        return valid_patient_ids

    def ctrl_test_indxs (self, df_raw):
        """Segments patient_ids into control and case groups and returns each group's ids.

        Segmentation is done by tagging patient_ids with `event` == 'positive bloodculture' as the `case_grp`.
        It follows that `control grp` is the set complement of the 'case_grp'.

        Parameters
        ----------
        df_raw : DataFrame
            The raw patient timeseries dataframe containing "patient_id" and "event" columns.

        Returns
        -------
        tuple[array, array]
            Patient_ids of the case cohort. Ids with event label "positive bloodculture".
            Patient_ids of the control cohort. The complement of the `test_ids` set.

        """
        assert all( col in df_raw.columns for col in ['measurement_datetime', 'event']), "Input df.columns must include ['measurement_datetime', 'event']."

        full_hist_testids = self.record_validity(df_raw, self.hist_length)
        full_hist_ctrlids = self.record_validity(df_raw, self.t_control)
        
        df_events = df_raw[['patient_id', 'measurement_datetime', 'event']].dropna(subset = ['event'])  # drop nans in every patient_id timeseries
        df_events = df_events.set_index('patient_id')

        df_events_test = df_events.loc[full_hist_testids]
        # sepsis_condition = df_events_test['event'].str.contains('positive')
        sepsis_condition = (df_events_test.event.str.contains('positive')) & (df_events_test.measurement_datetime >= self.hist_length)
        test_grp = df_events_test.loc[sepsis_condition].index.drop_duplicates()
        test_ids = test_grp.values

        df_events_ctrl = df_events.loc[full_hist_ctrlids]
        complement_indxs = df_events_ctrl.index.difference(test_ids)
        control_ids = complement_indxs.values


        return test_ids, control_ids


    @staticmethod
    def process_obs(obs):
        """Calculates and rounds the gestational age `ga` per patient. Returns df with `gender` & gestational age `ga` per patient.
        
        Creates a new column `ga`= round(`ga_weeks` + `ga_days`) per patient.
        `ga_days` >= 4 days are rounded up to 1 week, thus `ga_weeks` increases by 1.
        `gender` is converted to a categorical variable in the returned df.
            
        Parameters
        ----------
        obs : DataFrame
            Data frame with columns ['ga_weeks', 'ga_days', 'gender'].           
        
        Returns
        -------
        DataFrame
            A subset of `obs` df with 2 columns ['gender', 'ga'].
            
        """
        obs['ga_weeks'] = obs.ga_weeks.astype('uint16')
        obs['ga_days'] = obs['ga_days'].fillna(0).astype('uint16')
        obs['ga'] = obs.apply(lambda row: round((row.ga_weeks * 7 + row.ga_days) / 7) , axis = 1).astype('uint16')
        obs['gender'] = obs['gender'].astype('category')
        obs = obs.replace({'M': 'Male', 'F': 'Female'})
        return obs[['gender', 'ga']]
    
    def corrected_control_indxs(self, df_raw, obs):
        """Randomly select a control group while adjusting for the gestational age (ga) based on its distribution in the case group. Returns the indices of the ga adjusted control patients.
        
        Parameters
        ----------
        df_raw : DataFrame
            The raw patient timeseries dataframe containing "patient_id" and "event" columns.
            
        obs : DataFrame
            Data frame with columns ['ga_weeks', 'ga_days', 'gender'].    
        
        Returns
        -------
        tuple[array, array]
            array for the case patients.
            array for the contorl patient ids randomly selected based on the case group's ga distribution.
        """
        test_indxs, ctrl_indxs = self.ctrl_test_indxs(df_raw)
        obs_df = self.process_obs(obs)
        case_obs = obs.loc[test_indxs]
        control_obs = obs.loc[ctrl_indxs]
        case_ga_dist = case_obs.ga.value_counts().to_dict()
        case_gender_dist = case_obs.gender.value_counts().to_dict()
        
        
        randomizer = np.random.RandomState(self.seed)
        ga_corrected_ctrl_indxs = np.array([], dtype = 'uint16')
        for bin, freq in case_ga_dist.items():
            bin_indxs = control_obs.loc[obs.ga == bin].index
            bin_sample = randomizer.choice(bin_indxs, freq, replace = False)
            ga_corrected_ctrl_indxs = np.append(ga_corrected_ctrl_indxs, bin_sample)
            
        return test_indxs, ga_corrected_ctrl_indxs
  

    def ctrl_test_dfs(self, df_raw, obs):
        """Subsets the raw dataframe `df_raw` into test and control timeseries dataframe. Returns a tuple of 2 dfs: the case group and control group dfs.
        
        Creates the final case and control dfs. The distribution of gestational weeks is maintained in both groups.
        Raw physiological columns with NaNs >= 80% are discarded. Only patients with record lengths >= `hist_length` are selected. Returns separate dfs for the case and control groups.
        If `sample` > 0 in the class instance, a sub sample of the patient_ids is uniformaly drawn from `df_raw` and processed.
        
        Parameters
        ----------
        df_raw: DataFrame
            The raw patient timeseries dataframe with all columns.

        Returns
        -------
        tuple[DataFrame, DataFrame]
            The subset of the raw timeseries with the data points of the case group.
            The subset of the raw timeseries with the data points of the control group.

        """
        self.test_indxs, self.ctrl_indxs = self.corrected_control_indxs(df_raw, obs)
        test_ids = self.test_indxs
        control_ids = self.ctrl_indxs
        randomizer = np.random.RandomState(self.seed)
        
        if self.sample > 0:
            assert isinstance(self.sample, int), 'sample type must be an integer (default 0)'
            test_ids = randomizer.choice(self.test_indxs, self.sample, replace = False)
            control_ids = randomizer.choice(self.ctrl_indxs, self.sample, replace = False)
    
        df_raw = df_raw.set_index('patient_id')
        test_df = df_raw.loc[test_ids]
        test_nans = test_df.isna().sum() / len(test_df)
        effective_feats = test_nans.loc[test_nans < 0.8]
        effective_feats = effective_feats.drop(['minutes_since_birth'])
        effective_feats = effective_feats.index.insert(3, 'event')
        test_df = test_df[effective_feats]  # case df with effective features columns
        
        corrected_control_ids = randomizer.choice(control_ids, len(test_ids), replace = False)
        corrected_control_ids.sort()
        ctrl_df = df_raw.loc[corrected_control_ids, effective_feats] # control df with effective features columns


        return test_df, ctrl_df
    
    @staticmethod 
    def t_sepsis_timetable(case_df):
        """Scans the measurement_datetime column of the case group and records the timestamp of event `postive bloodculture` per patient id. Returns the dict {patient_id: t_sepsis}.
        
        Multiple postive culture events per patient are reduced to the earliest instance.
        
        Parameters
        ----------
        case_df : DataFrame
            Dataframe of case patients with patient_id as index and columns include ['measurement_datetime', 'event'].
            
        Returns
        -------
        dict
            dict with patient_ids as key and the patient's t_sepsis as value.
        
        """
        # Locate the t_sepsis timestamp per patient_id
        case_df = case_df.dropna(subset =['event'])
        t_sepsis = case_df.loc[case_df.event.str.contains('positive'), 'measurement_datetime']
        
        if t_sepsis.index.has_duplicates:
            t_sepsis = t_sepsis.loc[~t_sepsis.index.duplicated()] # if multiple positive timestamps, select the first instance 

        return t_sepsis
    
    def t_control_timetable(self, control_df):
        """Randomly assigns a t_control timestamp per control patient. Returns the dict {patient_id: t_control}.
        
        Selects a random patient_t_control timestamp from the closed interval [self.hist_length: self.t_control] such that:
        self.t_control < patient_t_control >= hist_lenght (ex: 24 hours < patient_t_control >= 60 hours).
                
        Parameters
        ----------
        control_df : DataFrame
            Dataframe of control patients with patient_id as index and columns include ['measurement_datetime'].
            
        Returns
        -------
        dict
            dict with patient_ids as key and the patient's random t_control as value.
        
        """
        randomizer = np.random.RandomState(self.seed)
        # Locate the t_sepsis timestamp per patient_id
        ctrl_indxs = control_df.index.unique()
        data = randomizer.randint(self.hist_length, self.t_control + 1, len(ctrl_indxs))
        control_lookup = pd.Series(data, ctrl_indxs)
        return control_lookup
    
    def subset_horizon(self, case_df, control_df):
        """Extracts the horizon segment of a patient record. Returns the case and control dataframes with the extracted horizon per patient.
        
        Using the t_sepsis and t_control timetable, each patient record in control and case dfs is subset to extract the segment of length self.horizon hours that directly preceeds the t_sepsis or t_control timestamp per patient.
                
        Parameters
        ----------
        case_df : DataFrame
            Dataframe of case patients with patient_id as index and columns include ['measurement_datetime'].
            
        control_df : DataFrame
            Dataframe of control patients with patient_id as index and columns include ['measurement_datetime'].
            
        Returns
        --------
        tuple[DataFrame, DataFrame]
            DataFrame of the extracted horizon per case patient.
            DataFrame of the extracted horizon per control patient.
        """
        t_sepsis_lookup = self.t_sepsis_timetable(case_df)
        t_control_lookup = self.t_control_timetable(control_df)
        slc = pd.IndexSlice
        
        case_df = case_df.set_index('measurement_datetime', append = True)
        case_df = case_df.sort_index(ascending= True)
        
        control_df = control_df.set_index('measurement_datetime', append = True)
        control_df = control_df.sort_index(ascending= True)
        
        case_horizon_df = pd.concat([case_df.loc[slc[patient, t_onset - self.horizon: t_onset], :] for patient, t_onset in t_sepsis_lookup.items()])
        control_horizon_df = pd.concat([control_df.loc[slc[patient, t_onset - self.horizon: t_onset], :] for patient, t_onset in t_control_lookup.items()])
        
        case_horizon_df.reset_index(level = 'measurement_datetime', inplace = True)
        control_horizon_df.reset_index(level = 'measurement_datetime', inplace = True)

        return case_horizon_df, control_horizon_df
    
    

class construct_features(process_raw):
    """Class to engineer statistical features from the patients' historical data.
    """
    def __init__(self, seed = 606, interval_length= 3):
        """Instantiates the feature constructor class with a sliding window length and a seed.
        """
        self.interval_length= interval_length
        super().__init__(seed)
    
    @staticmethod
    def feature_vector(df, patient, interval = 3):
        """Calculates the statistical feature per physiological marker per a single patient's horizon. Returns the statistical feature vector of the patient.
        
        The function slides a window with lenght = interval hours across a patient horizon of length self.horizon hours to obtain `self.horizon` // `interval` time segments (ex: 12h//3h = 4 time segments)
        Eight statistical features are calculated per window, generating 8 * `time segments` features per physiological marker.
        
        Parameters
        ----------
        patient : int
            Patient's index.
            
        interval : int
            Sliding window's length in hours.
            
        Returns
        -------
        DataFrame
            Df of the statistical feature vector of the input patient. Df.shape[1]= 8 stat feats * (horizon//window length) * num physio markers.
            
        """
        grouper= pd.Grouper( level= 'measurement_datetime', freq= f'{interval}H')
        
        for col in ['event', 'hours_since_birth']:
            if col in df.columns:
                df = df.drop(col, axis= 1)
            
        case_feats= df.groupby(by= grouper).agg(['mean', 'median', 'std', 'var', 'min', 'max', 'skew', pd.DataFrame.kurtosis])
        # age= df['hours_since_birth'].groupby(by= grouper).last()
        # case_feats.insert(0, ('hours_since_birth','age'), age)
        wide= case_feats.T.unstack()
        n_intervals= wide.columns.get_level_values(0).unique()
        wide.columns= wide.columns.map(lambda tup: f"Int_{n_intervals.get_loc(tup[0])}_{tup[1]}")
        wide= wide.unstack().to_frame().T
        return wide
    
    @staticmethod
    def encode_obs(obs):
        """Dummy encodes the gender variable in the observations data. Returns df with gender one-hot encoded.
        
        Parameters
        ----------
        obs : DataFrame
            Df of the patients profile including columns ['ga', 'gender'].
            
        Returns
        -------
        DataFrame
            The observations df with gender one-hot encoded.
        """
        gender_codes = pd.get_dummies(obs.gender)
        obs = pd.concat([obs.ga, gender_codes], axis = 1)
        return obs 
    
    def feature_df(self, df, obs):
        """Unifies the patient's statistical feature vectors and their profile features. Returns DataFrame with the complete feature vector per patient.
        
        Parameters
        ----------
        df : DataFrame
            Df containing the case or control patients.
            
        obs : DataFrame
            Df of the patients profile including columns ['ga', 'gender'].
            
        Returns
        -------
        DataFrame
            The feature df for all patients the input df.
        """

        df['measurement_datetime'] =  pd.to_timedelta(df.measurement_datetime, unit= 'm')
        df= df.set_index('measurement_datetime', append= True)
        feat_df= df.groupby('patient_id').apply(lambda x: self.feature_vector(x, x.name, self.interval_length) ).droplevel(1)
        
        obs = self.process_obs(obs)
        obs = self.encode_obs(obs)
        indx = pd.MultiIndex.from_product([['profile'], obs.columns.to_list()])
        obs.columns = indx
        
        feat_df = feat_df.merge(obs, right_index= True, left_index= True)
        
        return feat_df
    
    def modeling_set(self, case_features, control_features):
        """Propagates labels for case and control feature datasets. Returns case and control feature dfs with assigned label column.
        
        Parameters
        ----------
        case_features : DataFrame
            Df containing the feature vectors for the case patients.
            
        control_features : DataFrame
            Df containing the feature vectors for the control patients.
            
        Returns
        -------
        DataFrame
            The final unified df with feature vectors and labels for both control and case patients.
        """
        case_features['label'] = 1
        control_features['label'] = 0
        modeling_set = case_features.append(control_features, verify_integrity= True)
        modeling_set = modeling_set.sample(frac = 1, random_state = self.seed)  # resuffling dataframe
        
        return modeling_set
    
    @staticmethod
    def training_arrays(modeling_set):
        """Extracts from the modeling df X and y arrays for model training, containg the feature vectors and labels arrays respectively. Returns X and y numpy arrays.
        
        Parameters
        ----------
        modeling_set : DataFrame
            Df containing the feature vectors for the case patients.
            
        Returns
        -------
        tuple[array, array]
            Array containing the feature vector per patient.
            Array containing the labels vector per patient.
        """
        X_df = modeling_set.drop(columns= ['label'])
        y_df = modeling_set[['label']]
        X = X_df.values
        y = y_df.values.squeeze()
        
        return X, y
