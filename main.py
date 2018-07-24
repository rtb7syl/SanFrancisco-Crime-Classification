import numpy as np
import pandas as pd
import os
import pickle


from sklearn import preprocessing
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier




# load the training data

def load_data(datasets_dir):

    csv_path = os.path.join(datasets_dir,"train.csv")
    print('a')

    return pd.read_csv(csv_path)




def prepareRawData(train_data):
    # creating extra features like year,month ,day,hour,and minutes,by extracting data
    #from the Dates features.

    train_data['Dates'] = pd.to_datetime(train_data['Dates'])

    train_data['Year'] = train_data['Dates'].dt.year
    train_data['Month'] = train_data['Dates'].dt.month
    train_data['Day'] = train_data['Dates'].dt.day
    train_data['Hour'] = train_data['Dates'].dt.hour
    train_data['Minutes'] = train_data['Dates'].dt.minute

    print('Initial preparation of data done.......')
    return train_data






def one_hot_encoding_features(dataFrame,categoricalFeatures):

      #function to one-hot encode any number of categorical features of any dataframe
      # all the categorical features of the dataframe to be one-hot encoded must be in an array

      # returns an array of encoded features, each of which must be merged back to the original dataframe.



        
      #day_of_week_onehot = pd.get_dummies(crimes['DayOfWeek'])
      #day_of_week_onehot["Index"] = np.arange(878049)

      #pddistrict_onehot = pd.get_dummies(crimes['PdDistrict'])
      #pddistrict_onehot["Index"] = np.arange(878049)


      #crimes = crimes.drop(['PdDistrict','DayOfWeek'],axis=1,inplace=True)

      #crimes = pd.merge(crimes,pddistrict_onehot,on='Index')

      oneHotEncodedFeatures = []        

      for categoricalFeature in categoricalFeatures:

          categoricalFeatureOneHot = pd.get_dummies(dataFrame[categoricalFeature])
          categoricalFeatureOneHot["Index"] = np.arange(878049)
                                                  
          print('Encoding..........')


          oneHotEncodedFeatures.append(categoricalFeatureOneHot)


      print('Encoding done !!!')
      return oneHotEncodedFeatures


                                                        
                                                        
                                                        
                                                        







def transform_cyclical_features(dataFrame):

    #transform cyclical features like hour,minutes and day of any dataframe.
    # returns the transformed dataframe
                                                                
    dataFrame['hourfloat'] = dataFrame['Hour'] + dataFrame['Minutes']/60.0
                                                                
    dataFrame['hourfloat_sine']= np.sin(dataFrame.hourfloat*(2.*np.pi/24.))
    dataFrame['hourfloat_cosine']= np.cos(dataFrame.hourfloat*(2.*np.pi/24.))
                                                                                
    dataFrame['month_sine']= np.sin((dataFrame['Month'] - 1)*(2.*np.pi/12.))
    dataFrame['month_cosine']= np.cos((dataFrame['Month'] - 1)*(2.*np.pi/12.))
                                                                                            
    dataFrame['day_sine']= np.sin((dataFrame['Day'] - 1)*(2.*np.pi/31.))
    dataFrame['day_cosine']= np.cos((dataFrame['Day'] - 1)*(2.*np.pi/31.))                                                                                                    
                                            
    print('Cyclical features transformed into sine and cosine!!!')

    return dataFrame



# functions for constructing different kinds of label vectors for different kinds of predictions.
# Numbered as #1l, #2l ,#3l and so on
 
def label_type1l(crimesCategory):

    #Violent/Non-violent crimes prediction

    #takes in the pandas series format of the raw Category column.
    #returns a numpy array of the label vector


    violent_crimes = ['LARCENY/THEFT','ASSAULT','DRUG/NARCOTIC','VANDALISM','BURGLARY',
                              
                              'ROBBERY','WEAPON LAWS','SEX OFFENSES FORCIBLE',
                                                
                                                'KIDNAPPING','ARSON','SEX OFFENSES NON FORCIBLE']


    crimesCategory = np.where(crimesCategory.isin(violent_crimes), 1,0)

    y_data = crimesCategory

    print('Labels transformation done......',y_data)



    return y_data


def label_type2l(crimesCategory): #labels/targets are one-hot encoded

    #39 different types of crimes prediction (multi-class classification)

    #takes in the pandas series format of the raw Category column.
    #returns a numpy array of the label vector


    categoricalTargetsOneHot = pd.get_dummies(crimesCategory)


    #crimesCategory = np.where(crimesCategory.isin(violent_crimes), 1,0)

    y_data = categoricalTargetsOneHot.values

    print('Labels transformation done......',y_data)



    return y_data




def label_type3l(crimesCategory): #labels/targets are label encoded

    #39 different types of crimes prediction (multi-class classification)

    #takes in the pandas series format of the raw Category column.
    #returns a numpy array of the label vector

    le = preprocessing.LabelEncoder()

    y_data = le.fit_transform(crimesCategory)


    

    #y_data = categoricalTargetsOneHot.values

    print('Labels transformation done......',y_data)



    return y_data


def countFeatureLoadFromPickle():

    df = pd.read_pickle('countfeature.pickle',compression='gzip')

    return df

    

def countFeatureTransform(crimesFeatures,crimesCategory,feature_to_be_transformed):

    featureUniqueCategories = crimesFeatures[feature_to_be_transformed].unique()

    print(featureUniqueCategories)

    crimesCat = label_type1l(crimesCategory)



    #prior_freq_1 = number_of_1s/878049

    #prior_freq_0 = number_of_0s/878049

    crimesFeatures["Cat"] = crimesCat

    crimesFeatures["num_class_00"] = 0
    crimesFeatures["num_class_01"] = 0

    crimesFeatures["freq_class_00"] = 0
    crimesFeatures["freq_class_01"] = 0

    #crimesFeatures["class_00_logOdds"] = 0
    #crimesFeatures["class_01_logOdds"] = 0


    for featureUniqueCategory in featureUniqueCategories:

        featureUniqueCategoryDict = crimesFeatures.loc[crimesFeatures[feature_to_be_transformed] == featureUniqueCategory]['Cat'].value_counts().to_dict()
        totalUniqueCatInstances = crimesFeatures.loc[crimesFeatures[feature_to_be_transformed] == featureUniqueCategory]['Cat'].value_counts().sum()


        print(featureUniqueCategory,featureUniqueCategoryDict,totalUniqueCatInstances)

        cat_0 = featureUniqueCategoryDict.get(0,0)

        cat_1 = featureUniqueCategoryDict.get(1,0)

        freq_0 = cat_0/totalUniqueCatInstances

        freq_1 = cat_1/totalUniqueCatInstances


        crimesFeatures.loc[crimesFeatures[feature_to_be_transformed] == featureUniqueCategory,'num_class_00'] = cat_0

        crimesFeatures.loc[crimesFeatures[feature_to_be_transformed] == featureUniqueCategory,'freq_class_00'] = freq_0

        crimesFeatures.loc[crimesFeatures[feature_to_be_transformed] == featureUniqueCategory,'freq_class_01'] = freq_1


        crimesFeatures.loc[crimesFeatures[feature_to_be_transformed] == featureUniqueCategory,'num_class_01'] = cat_1 

    print('Count feature transforming done.....')

    print(crimesFeatures['num_class_00'])
    print(crimesFeatures['num_class_01'])
    print(crimesFeatures['freq_class_00'])
    print(crimesFeatures['freq_class_01'])

    #outfile = open(filename,'wb')
    crimesFeatures.to_pickle('countfeature.pickle', compression='gzip')

    return crimesFeatures








# some combinations of features ,going to be used . 
# Numbered as # 1f, # 2f, # 3f etc


# #1f feature combination
def features_set1f(crimesFeatures):

    # function takes in a data frame consisting only of the features,not the label.
    # returns a numpy 2D array format of all the feature vectors



    crimesFeatures['Index'] = np.arange(878049)

    categoricalFeatures = ['PdDistrict','DayOfWeek']



    oneHotEncodedFeatures = one_hot_encoding_features(crimesFeatures,categoricalFeatures)

    print('Categorical features oneHot encoding done.....')



    for oneHotEncodedFeature in oneHotEncodedFeatures:

        crimesFeatures = pd.merge(crimesFeatures,oneHotEncodedFeature,on='Index')

        print('Merging oneHot features.....')

    print('Merging oneHot features done....')


    crimesFeatures = transform_cyclical_features(crimesFeatures)
    print('Transforming cyclical features done....')


    crimesFeatures.drop(

                        ['Dates', 'Descript','Address','Resolution',
                        'Month','Day','Hour','Minutes',
                        'PdDistrict','DayOfWeek','hourfloat','Index'],

                          axis=1,inplace=True)


    print('All features tranformation done.....')

    print(crimesFeatures.head())
    print(crimesFeatures.info())
    print(crimesFeatures.shape)






    # converts dataframe to numpy array
    X_data = crimesFeatures.values


    return X_data








# #2f feature combination
def features_set2f(crimesFeatures):

    # function takes in a data frame consisting only of the features,not the label.
    # returns a numpy 2D array format of all the feature vectors




    crimesFeatures['Index'] = np.arange(878049)

    categoricalFeatures = ['PdDistrict','DayOfWeek','Month','Hour','Minutes']



    oneHotEncodedFeatures = one_hot_encoding_features(crimesFeatures,categoricalFeatures)

    print('Categorical features oneHot encoding done.....')



    for oneHotEncodedFeature in oneHotEncodedFeatures:

        crimesFeatures = pd.merge(crimesFeatures,oneHotEncodedFeature,on='Index')

        print('Merging oneHot features.....')

    print('Merging oneHot features done....')


    crimesFeatures = transform_cyclical_features(crimesFeatures)
    print('Transforming cyclical features done....')


    crimesFeatures.drop(

                        ['Dates', 'Descript','Address','Resolution',
                        'Month','Day','Hour','Minutes',
                        'PdDistrict','DayOfWeek','Index','hourfloat'],

                          axis=1,inplace=True)


    print('All features tranformation done.....')

    print(crimesFeatures.head())
    crimesFeatures.info()
    print(crimesFeatures.shape)






    # converts dataframe to numpy array
    X_data = crimesFeatures.values

    print(X_data)
    return X_data



# #3f feature combination
def features_set3f(crimesFeatures,crimesCategory):

    # function takes in a data frame consisting only of the features,not the label.
    # returns a numpy 2D array format of all the feature vectors



    crimesFeatures['Index'] = np.arange(878049)

    crimesFeatures = countFeatureTransform(crimesFeatures,crimesCategory,"Address")
    
    categoricalFeatures = ['PdDistrict','DayOfWeek']

    

    oneHotEncodedFeatures = one_hot_encoding_features(crimesFeatures,categoricalFeatures)

    print('Categorical features oneHot encoding done.....')



    for oneHotEncodedFeature in oneHotEncodedFeatures:

        crimesFeatures = pd.merge(crimesFeatures,oneHotEncodedFeature,on='Index')

        print('Merging oneHot features.....')

    print('Merging oneHot features done....')


    #crimesFeatures = transform_cyclical_features(crimesFeatures)
    #print('Transforming cyclical features done....')


    


    crimesFeatures.drop(

                        ['Dates', 'Descript','Address','Resolution',
                         'Day','PdDistrict','DayOfWeek','Index','Cat'],
                        
                        axis=1,inplace=True)


    print('All features tranformation done.....')

    print(crimesFeatures.head())
    crimesFeatures.info()
    print(crimesFeatures.shape)






    # converts dataframe to numpy array
    X_data = crimesFeatures.values


    return X_data



# #4f feature combination
def features_set4f(crimesFeatures): # only PdDistrict and DayOfWeek are one-hot encoded

    # function takes in a data frame consisting only of the features,not the label.
    # returns a numpy 2D array format of all the feature vectors



    crimesFeatures['Index'] = np.arange(878049)

    categoricalFeatures = ['PdDistrict','DayOfWeek']



    oneHotEncodedFeatures = one_hot_encoding_features(crimesFeatures,categoricalFeatures)

    print('Categorical features oneHot encoding done.....')



    for oneHotEncodedFeature in oneHotEncodedFeatures:

        crimesFeatures = pd.merge(crimesFeatures,oneHotEncodedFeature,on='Index')

        print('Merging oneHot features.....')

    print('Merging oneHot features done....')


    #crimesFeatures = transform_cyclical_features(crimesFeatures)
    #print('Transforming cyclical features done....')


    crimesFeatures.drop(

                        ['Dates', 'Descript','Address','Resolution',
                        'Day',
                        'PdDistrict','DayOfWeek','Index'],

                          axis=1,inplace=True)


    print('All features tranformation done.....')

    print(crimesFeatures.head())
    crimesFeatures.info()
    print(crimesFeatures.shape)






    # converts dataframe to numpy array
    X_data = crimesFeatures.values


    return X_data





def splitDataIntoFeaturesAndLabels(crimes):
    print('splitting started.....')
    crimesCategory = crimes['Category']

    crimesFeatures = crimes.drop('Category',axis = 1)


    X_data = features_set3f(crimesFeatures,crimesCategory)
    y_data = label_type1l(crimesCategory)

    print('Splitting done.....')

    return (X_data,y_data)





def kFoldValidation(X_data,y_data,n_splits,shuffle=True):


    kf = KFold(n_splits=n_splits,shuffle=shuffle)

    kf.get_n_splits(X_data)




    print(kf)

    print('Hey there,hang on quite a bit...Main jobs started...Come back later!!')


    i = 1

    train_acc = []
    val_acc = []

    for train_index, test_index in kf.split(X_data):
            
      print('*' * 15)
                
      print(' Training and validating ' + str(i) + 'th fold ..........')
                        
      print("TRAIN INDICES:", train_index, "        TEST INDICES:", test_index)

      X_train, X_val = X_data[train_index], X_data[test_index]
                                    
      y_train, y_val = y_data[train_index], y_data[test_index]
                                            
      #clf =  RandomForestClassifier(n_estimators=200,max_depth=15)
      clf = KNeighborsClassifier(n_neighbors = 500)

      clf.fit(X_train, y_train)
      

      val_acc_each_fold = clf.score(X_val,y_val)

      train_acc_each_fold = clf.score(X_train,y_train)
                                                                
      print('Train Accuracy for '+ str(i) + 'th fold  =  ',train_acc_each_fold)
                                                                        
      print('Validation Accuracy for ' + str(i) + 'th fold  =  ',val_acc_each_fold)
                                                                            
      train_acc.append(train_acc_each_fold)

      val_acc.append(val_acc_each_fold)
                                                                                            
      i = i + 1
                                                                                                    
                                                      
    print(train_acc)
    print(val_acc)
    mean_fold_train_acc = sum(train_acc)/len(train_acc)
                                                                                                    
    mean_fold_val_acc = sum(val_acc)/len(val_acc)
    print('Overall avg train acc = ',mean_fold_train_acc)
    print('Overall avg val acc = ',mean_fold_val_acc)



def main(datasets_dir):

    print('All jobs started.....')
      

    train_data = load_data(datasets_dir)


    train_data = prepareRawData(train_data)




    #preserving the train_data ,and copying it into
    #crimes Dataframe

    crimes = train_data.copy()

    X_data,y_data = splitDataIntoFeaturesAndLabels(crimes)

    print(X_data,y_data)

    kFoldValidation(X_data,y_data,n_splits=4,shuffle=True)

    print('All jobs complete......')





main('/home/rtb7syl/ai_ml_dl_projects_codebase/sf_crime_classification/datasets/all')


