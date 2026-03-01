# module imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
from sklearn.decomposition import PCA

# model imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA


# processing imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score,classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

np.random.seed(1635848)

def process(DoS_only=True):
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
    ]

    # Load specific URLs for NSL-KDD Train and Test
    train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
    test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

    print("Downloading and loading data...")
    df_train = pd.read_csv(train_url, names=columns)
    df_test = pd.read_csv(test_url, names=columns)

    mappings = {
        'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
        'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2,
        'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
        'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4
    }

    df_train["attack"] = df_train["label"].map(mappings)
    df_test["attack"] = df_test["label"].map(mappings)
    
    df_train.drop("label", axis=1, inplace=True)
    df_test.drop("label", axis=1, inplace=True)

    df_train.rename({"attack": "label"}, axis=1, inplace=True)
    df_test.rename({"attack": "label"}, axis=1, inplace=True)
    
    if DoS_only:
        to_drop_DoS = [2,3,4]
        DoS_train_df = df_train[~df_train['label'].isin(to_drop_DoS)]
        DoS_test_df = df_test[~df_test['label'].isin(to_drop_DoS)]
        
        return DoS_train_df, DoS_test_df

    return df_train, df_test


def load_unsw():
    train = pd.read_csv("unsw/Training and Testing Sets/UNSW_NB15_training-set.csv").iloc[:, 1:]
    test = pd.read_csv("unsw/Training and Testing Sets/UNSW_NB15_testing-set.csv").iloc[:, 1:]

    train = pd.concat([train[train["attack_cat"] == "Normal"], train[train["attack_cat"] == "Generic"]])
    test = pd.concat([test[test["attack_cat"] == "Normal"], test[test["attack_cat"] == "Generic"]])

    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    # label_encoders = {}  
    # for col in train.select_dtypes(include='object').columns:
    #     le = LabelEncoder()
    #     le.fit(pd.concat([train[col], test[col]]))
    #     train[col] = le.transform(train[col])
    #     test[col] = le.transform(test[col])
    #     label_encoders[col] = le

    X_train, y_train = train.drop(["label"], axis=1), train["label"].to_frame()
    X_test, y_test = test.drop(["label"], axis=1), test["label"].to_frame()

    X_train = X_train.drop(["attack_cat"], axis=1)
    X_test = X_test.drop(["attack_cat"], axis=1)

    # scaler = StandardScaler()
    # X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    # X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_test, y_train, y_test

# X_train,_,_,_ = load_unsw()
# print(X_train.head())


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# def process(DoS_only=False):
#     # ---------------------------------------------------------
#     # 1. Load Data and Define Headers
#     # ---------------------------------------------------------
#     file_path_full_training_set = 'nsl-kdd/KDDTrain+.txt'
#     file_path_test = 'nsl-kdd/KDDTest+.txt'

#     df = pd.read_csv(file_path_full_training_set)
#     test_df = pd.read_csv(file_path_test)

#     columns = [
#         'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
#         'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 
#         'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 
#         'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
#         'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 
#         'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
#         'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
#         'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
#         'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
#         'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
#     ]

#     df.columns = columns
#     test_df.columns = columns

#     # ---------------------------------------------------------
#     # 2. Categorical Column Encoding (Label + OneHot)
#     # ---------------------------------------------------------
#     categorical_columns = ['protocol_type', 'service', 'flag']
    
#     # Isolate categorical data
#     df_categorical_values = df[categorical_columns]
#     testdf_categorical_values = test_df[categorical_columns]

#     # --- Generate Custom Column Names (Manual Sorting) ---
#     # Protocol
#     unique_protocol = sorted(df.protocol_type.unique())
#     unique_protocol2 = ['Protocol_type_' + x for x in unique_protocol]
    
#     # Service (Train)
#     unique_service = sorted(df.service.unique())
#     unique_service2 = ['service_' + x for x in unique_service]
    
#     # Service (Test) - Note: Test set might have different services
#     unique_service_test = sorted(test_df.service.unique())
#     unique_service2_test = ['service_' + x for x in unique_service_test]
    
#     # Flag
#     unique_flag = sorted(df.flag.unique())
#     unique_flag2 = ['flag_' + x for x in unique_flag]

#     # Assemble Column Lists
#     dumcols = unique_protocol2 + unique_service2 + unique_flag2
#     testdumcols = unique_protocol2 + unique_service2_test + unique_flag2

#     # --- Apply Label Encoder ---
#     # Note: Applying separately to train and test as per original logic
#     df_categorical_values_enc = df_categorical_values.apply(LabelEncoder().fit_transform)
#     testdf_categorical_values_enc = testdf_categorical_values.apply(LabelEncoder().fit_transform)

#     # --- Apply OneHot Encoder ---
#     enc = OneHotEncoder()
    
#     # Train
#     df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
#     df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(), columns=dumcols)
    
#     # Test
#     testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
#     testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(), columns=testdumcols)

#     # ---------------------------------------------------------
#     # 3. Align Train/Test Columns (Handle Missing Services)
#     # ---------------------------------------------------------
#     trainservice = df['service'].tolist()
#     testservice = test_df['service'].tolist()
    
#     # Find services present in Train but missing in Test
#     difference = list(set(trainservice) - set(testservice))
#     difference_cols = ['service_' + x for x in difference]

#     # Add missing columns to Test dataset with 0s
#     for col in difference_cols:
#         testdf_cat_data[col] = 0

#     # ---------------------------------------------------------
#     # 4. Join Encoded Data and Cleanup
#     # ---------------------------------------------------------
#     newdf = df.join(df_cat_data)
#     newdf.drop(['flag', 'protocol_type', 'service'], axis=1, inplace=True)

#     newdf_test = test_df.join(testdf_cat_data)
#     newdf_test.drop(['flag', 'protocol_type', 'service'], axis=1, inplace=True)

#     # ---------------------------------------------------------
#     # 5. Label Mapping
#     # ---------------------------------------------------------
#     label_mapping = {
#         'normal': 0,
#         # DoS (Class 1)
#         'neptune': 1, 'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1, 'mailbomb': 1, 
#         'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
#         # Probe (Class 2)
#         'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2, 'mscan': 2, 'saint': 2,
#         # R2L (Class 3)
#         'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3, 'phf': 3, 'spy': 3, 
#         'warezclient': 3, 'warezmaster': 3, 'sendmail': 3, 'named': 3, 'snmpgetattack': 3, 
#         'snmpguess': 3, 'xlock': 3, 'xsnoop': 3, 'httptunnel': 3,
#         # U2R (Class 4)
#         'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4, 'ps': 4, 'sqlattack': 4, 'xterm': 4
#     }

#     newdf['attack'] = newdf['attack'].map(label_mapping)
#     newdf_test['attack'] = newdf_test['attack'].map(label_mapping)

#     # ---------------------------------------------------------
#     # 6. Dataset Splitting by Attack Category
#     # ---------------------------------------------------------
#     to_drop_DoS = [2, 3, 4]
#     to_drop_Probe = [1, 3, 4]
#     to_drop_R2L = [1, 2, 4]
#     to_drop_U2R = [1, 2, 3]

#     # Create Subsets - Train
#     DoS_df = newdf[~newdf['attack'].isin(to_drop_DoS)]
#     Probe_df = newdf[~newdf['attack'].isin(to_drop_Probe)]
#     R2L_df = newdf[~newdf['attack'].isin(to_drop_R2L)]
#     U2R_df = newdf[~newdf['attack'].isin(to_drop_U2R)]

#     # Create Subsets - Test
#     DoS_df_test = newdf_test[~newdf_test['attack'].isin(to_drop_DoS)]
#     Probe_df_test = newdf_test[~newdf_test['attack'].isin(to_drop_Probe)]
#     R2L_df_test = newdf_test[~newdf_test['attack'].isin(to_drop_R2L)]
#     U2R_df_test = newdf_test[~newdf_test['attack'].isin(to_drop_U2R)]

#     # ---------------------------------------------------------
#     # 7. Final Return Logic
#     # ---------------------------------------------------------
#     if DoS_only:
#         return DoS_df, DoS_df_test
#     else:
#         # Concatenate all subsets (Note: This logic replicates 'normal' rows)
#         df_train = pd.concat([DoS_df, Probe_df, R2L_df, U2R_df], axis=0)
#         df_test = pd.concat([DoS_df_test, Probe_df_test, R2L_df_test, U2R_df_test], axis=0)
#         return df_train, df_test





if __name__ == "__main__":
    train, test = process(DoS_only=True)
    print(train.shape, test.shape)

# def process(DoS_only=False):
#     # fetch the training file
#     file_path_full_training_set = 'nsl-kdd/KDDTrain+.txt'
#     file_path_test = 'nsl-kdd/KDDTest+.txt'

#     #df = pd.read_csv(file_path_20_percent)
#     df = pd.read_csv(file_path_full_training_set)
#     test_df = pd.read_csv(file_path_test)

#     # add the column labels
#     columns = (['duration'
#     ,'protocol_type'
#     ,'service'
#     ,'flag'
#     ,'src_bytes'
#     ,'dst_bytes'
#     ,'land'
#     ,'wrong_fragment'
#     ,'urgent'
#     ,'hot'
#     ,'num_failed_logins'
#     ,'logged_in'
#     ,'num_compromised'
#     ,'root_shell'
#     ,'su_attempted'
#     ,'num_root'
#     ,'num_file_creations'
#     ,'num_shells'
#     ,'num_access_files'
#     ,'num_outbound_cmds'
#     ,'is_host_login'
#     ,'is_guest_login'
#     ,'count'
#     ,'srv_count'
#     ,'serror_rate'
#     ,'srv_serror_rate'
#     ,'rerror_rate'
#     ,'srv_rerror_rate'
#     ,'same_srv_rate'
#     ,'diff_srv_rate'
#     ,'srv_diff_host_rate'
#     ,'dst_host_count'
#     ,'dst_host_srv_count'
#     ,'dst_host_same_srv_rate'
#     ,'dst_host_diff_srv_rate'
#     ,'dst_host_same_src_port_rate'
#     ,'dst_host_srv_diff_host_rate'
#     ,'dst_host_serror_rate'
#     ,'dst_host_srv_serror_rate'
#     ,'dst_host_rerror_rate'
#     ,'dst_host_srv_rerror_rate'
#     ,'attack'
#     ,'level'])

#     df.columns = columns
#     test_df.columns = columns
#     pd.set_option('display.max_columns', 43)
#     df.head()

#     pd.set_option('display.max_rows', 23)
#     # print('Label distribution Training set:')
#     # print(df['attack'].value_counts())


#     # colums that are categorical and not binary yet: protocol_type (column 2), service (column 3), flag (column 4).
#     # explore categorical features
#     # print('Training set:')
#     for col_name in df.columns:
#         if df[col_name].dtypes == 'object' :
#             unique_cat = len(df[col_name].unique())
#             # print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))



#     # Test set
#     # print('Test set:')
#     for col_name in test_df.columns:
#         if test_df[col_name].dtypes == 'object' :
#             unique_cat = len(test_df[col_name].unique())
#             # print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

    
#     from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#     categorical_columns=['protocol_type', 'service', 'flag']
#     # insert code to get a list of categorical columns into a variable, categorical_columns
#     categorical_columns=['protocol_type', 'service', 'flag']
#     # Get the categorical values into a 2D numpy array
#     df_categorical_values = df[categorical_columns]
#     testdf_categorical_values = test_df[categorical_columns]
#     df_categorical_values.head()       

#     # protocol type
#     unique_protocol=sorted(df.protocol_type.unique())
#     string1 = 'Protocol_type_'
#     unique_protocol2=[string1 + x for x in unique_protocol]
#     # service
#     unique_service=sorted(df.service.unique())
#     string2 = 'service_'
#     unique_service2=[string2 + x for x in unique_service]
#     # flag
#     unique_flag=sorted(df.flag.unique())
#     string3 = 'flag_'
#     unique_flag2=[string3 + x for x in unique_flag]
#     # put together
#     dumcols=unique_protocol2 + unique_service2 + unique_flag2
#     # print(dumcols)

#     #do same for test set
#     unique_service_test=sorted(test_df.service.unique())
#     unique_service2_test=[string2 + x for x in unique_service_test]
#     testdumcols=unique_protocol2 + unique_service2_test + unique_flag2


#     df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)
#     # print(df_categorical_values_enc.head())
#     # test set
#     testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)

#     enc = OneHotEncoder()
#     df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
#     df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)
#     # df_cat_data = df_categorical_values_enc
#     # test set
#     testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
#     testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)
#     # testdf_cat_data = testdf_categorical_values_enc

#     df_cat_data.head()

#     trainservice=df['service'].tolist()
#     testservice= test_df['service'].tolist()
#     difference=list(set(trainservice) - set(testservice))
#     string = 'service_'
#     difference=[string + x for x in difference]

#     for col in difference:
#         testdf_cat_data[col] = 0

#     newdf=df.join(df_cat_data)
#     newdf.drop('flag', axis=1, inplace=True)
#     newdf.drop('protocol_type', axis=1, inplace=True)
#     newdf.drop('service', axis=1, inplace=True)
#     # test data
#     newdf_test=test_df.join(testdf_cat_data)
#     newdf_test.drop('flag', axis=1, inplace=True)
#     newdf_test.drop('protocol_type', axis=1, inplace=True)
#     newdf_test.drop('service', axis=1, inplace=True)
#     # print(newdf.shape)
#     # print(newdf_test.shape)

#     # take label column
#     labeldf=newdf['attack']
#     labeldf_test=newdf_test['attack']
#     # change the label column
#     newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
#                             'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
#                             ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
#                             'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
#     newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
#                             'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
#                             ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
#                             'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
#     # put the new label column back
#     newdf['attack'] = newlabeldf
#     newdf_test['attack'] = newlabeldf_test
#     # print(newdf['attack'].head())


#     to_drop_DoS = [2,3,4]
#     to_drop_Probe = [1,3,4]
#     to_drop_R2L = [1,2,4]
#     to_drop_U2R = [1,2,3]
#     DoS_df=newdf[~newdf['attack'].isin(to_drop_DoS)];
#     Probe_df=newdf[~newdf['attack'].isin(to_drop_Probe)];
#     R2L_df=newdf[~newdf['attack'].isin(to_drop_R2L)];
#     U2R_df=newdf[~newdf['attack'].isin(to_drop_U2R)];

#     #test
#     DoS_df_test=newdf_test[~newdf_test['attack'].isin(to_drop_DoS)];
#     Probe_df_test=newdf_test[~newdf_test['attack'].isin(to_drop_Probe)];
#     R2L_df_test=newdf_test[~newdf_test['attack'].isin(to_drop_R2L)];
#     U2R_df_test=newdf_test[~newdf_test['attack'].isin(to_drop_U2R)];

#     colNames=DoS_df.columns
#     colNames_test=DoS_df_test.columns

    
#     scaler1 = preprocessing.StandardScaler().fit(DoS_df)
#     DoS_df=scaler1.transform(DoS_df)
#     scaler2 = preprocessing.StandardScaler().fit(Probe_df)
#     Probe_df=scaler2.transform(Probe_df)
#     scaler3 = preprocessing.StandardScaler().fit(R2L_df)
#     R2L_df=scaler3.transform(R2L_df)
#     scaler4 = preprocessing.StandardScaler().fit(U2R_df)
#     U2R_df=scaler4.transform(U2R_df)
#     # test data
#     scaler5 = preprocessing.StandardScaler().fit(DoS_df_test)
#     DoS_df_test=scaler5.transform(DoS_df_test)
#     scaler6 = preprocessing.StandardScaler().fit(Probe_df_test)
#     Probe_df_test=scaler6.transform(Probe_df_test)
#     scaler7 = preprocessing.StandardScaler().fit(R2L_df_test)
#     R2L_df_test=scaler7.transform(R2L_df_test)
#     scaler8 = preprocessing.StandardScaler().fit(U2R_df_test)
#     U2R_df_test=scaler8.transform(U2R_df_test)


#     if not DoS_only:
#         df_train = pd.concat([DoS_df, Probe_df, R2L_df, U2R_df], axis=0)
#         df_test = pd.concat([DoS_df_test, Probe_df_test, R2L_df_test, U2R_df_test], axis=0)
#         return df_train, df_test
    
#     df_train = DoS_df
#     df_test = DoS_df_test

#     df_train = pd.DataFrame(df_train, columns=colNames)
#     df_test = pd.DataFrame(df_test, columns=colNames_test)

#     print(df_train.shape, df_test.shape)
#     return df_train, df_test


# # process(DoS_only=True)