import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from datetime import date
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score



class FeatureEngineering:

    def data_handling(self, df):
        enc = LabelEncoder()
        for i in range(df.shape[1]):
            if df.iloc[:,i].isna().sum()!=0:
                df.iloc[:,i].fillna(df.iloc[:,i].mean(), inplace=True)
                print(df.iloc[:,i].isna().sum())

            if df.iloc[:,i].dtype == object:
                print("Column contains a categorical value: ",i)
                df.iloc[:,i] = enc.fit_transform(df.iloc[:,i])
                print('Categories: ',df.iloc[:,i].unique())
        return df



    #Imputation

    def impute_mean(self,df):
        for i in range(df.shape[1]):
            if df.iloc[:,i].isna().sum()!=0:
                df.iloc[:,i].fillna(df.iloc[:,i].mean(), inplace=True)
                print(df.iloc[:,i].isna().sum())
        return df

    def impute_mode(self,df):
        for i in range(df.shape[1]):
            if df.iloc[:,i].isna().sum()!=0:
                df.iloc[:,i].fillna(df.iloc[:,i].mode()[0], inplace=True)
                print(df.iloc[:,i].isna().sum())
        return df

    def impute_median(self,df,col):
        print("Column contains NA value : " + str(col))
        df.iloc[:,col].fillna(df.iloc[:,col].median(), inplace=True)
    
    def impute_random(self,df,col):
        print("Column contains NA value : " + str(col))
        random_sample = df.iloc[:,col].dropna().sample(df.iloc[:,col].isna().sum(),random_state = 0)
        random_sample.index = df[df.iloc[:,col].isna()].index
        df.loc[df.iloc[:,col].isna(),df.columns[col]] = random_sample
    
  
    def impute_end_dist(self,df,col):
        print("Column contains NA value : " + str(col))
        extreme = df.iloc[:,col].mean() + 3* df.iloc[:,col].std()
        df.iloc[:,col] = df.iloc[:,col].fillna(extreme)

    #LabelEncoding
    def label_enc(self,df):
        enc = LabelEncoder()
        for i in range(df.shape[1]):
            if df.iloc[:,i].dtype == object:
                print("Column contains a categorical value: ",i)
                df.iloc[:,i] = enc.fit_transform(df.iloc[:,i])
                print('Categories: ',df.iloc[:,i].unique())
        return df






   

    #Handling Outliers
    def outliers_std(self,df,col,factor): #same as standf[df.columns[col]] = df[(df[df.columns[col]]<ub) & (df[df.columns[col]]>lb)]dard-scaler
        ub = df.iloc[:,col].mean() + factor * df.iloc[:,col].std()
        lb = df.iloc[:,col].mean() - factor * df.iloc[:,col].std()
        df[df.columns[col]] = df[(df[df.columns[col]]<ub) & (df[df.columns[col]]>lb)]

    def outliers_percentile(self,df,col):
        ub = df.iloc[:,col].quantile(0.95)
        lb = df.iloc[:,col].quantile(0.05)
        df[df.columns[col]] = df[(df[df.columns[col]]<ub) & (df[df.columns[col]]>lb)]

    #Normalization
    def min_max(self,df,col):
        df_min = df.iloc[:,col].min()
        df_max = df.iloc[:,col].max()

        df.iloc[:,col] = (df.iloc[:,col] - df_min) / (df_max - df_min)

    def std_scal(self,df,col):
        df_mean = df.iloc[:,col].mean()
        df_std = df.iloc[:,col].std()

        df.iloc[:,col] = (df.iloc[:,col] - df_mean)/df_std

    def avg_scal(self,df,col):
        df_mean = df.iloc[:,col].mean()
        df_max = df.iloc[:,col].max()

        df.iloc[:,col] = (df.iloc[:,col] - df_mean) / (df_max - df_mean)
    
    #Discritization
    def binning_uniform(self,data,n_bins):
        obj = KBinsDiscretizer(n_bins=n_bins , encode='ordinal',strategy='uniform')
        data = obj.fit_transform(data)
        return data

    def binning_quantile(self,data,n_bins):
        obj = KBinsDiscretizer(n_bins=n_bins , encode='ordinal',strategy='quantile')
        data = obj.fit_transform(data)
        return data

    def binning_kmeans(self,data,n_bins):
        obj = KBinsDiscretizer(n_bins=n_bins , encode='ordinal',strategy='kmeans')
        data = obj.fit_transform(data)
        return data

    #Feature Selection    

    def factor_analysis(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        fa_obj = FactorAnalysis(n_components=n_components,random_state=0)
        fa_obj.fit(scaled_data)
        x = fa_obj.transform(scaled_data)
        return x


    def pca(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        pca_obj = PCA(n_components=n_components)
        pca_obj.fit(scaled_data)
        x_pca = pca_obj.transform(scaled_data)
        return x_pca

    def fast_ica(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        fi_obj = FastICA(n_components=n_components,random_state=0)
        fi_obj.fit(scaled_data)
        x = fi_obj.transform(scaled_data)
        return x
   
    def incremental_pca(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = IncrementalPCA(n_components=n_components)
        obj.fit(scaled_data)
        x = obj.transform(scaled_data)
        return x
        

    def kernel_pca(self,data,n_components,kernel='linear'): #kernel = ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = KernelPCA(n_components=n_components,kernel=kernel)
        obj.fit(scaled_data)
        x = obj.transform(scaled_data)
        return x


    def lda(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = LatentDirichletAllocation(n_components=n_components,random_state=0)
        obj.fit(scaled_data)
        x = obj.transform(scaled_data)
        return x

    def mini_batch_dict_learning(self,data,n_components,transform_algorithm= 'lasso_lars'): #transform_algorithm = 'lasso_lars','lasso_cd'
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = MiniBatchDictionaryLearning(n_components=n_components,transform_algorithm=transform_algorithm)
        obj.fit(scaled_data)
        x = obj.transform(scaled_data)
        return x
        
    def mini_batch_sparse_pca(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = MiniBatchSparsePCA(n_components=n_components,random_state=0)
        obj.fit(scaled_data)
        x = obj.transform(scaled_data)
        return x

    def nmf(self,data,n_components,init='random'): #init{‘random’, ‘nndsvd’, ‘nndsvda’, ‘nndsvdar’, ‘custom’}
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = NMF(n_components=n_components, init=init, random_state=0)
        obj.fit(scaled_data)
        x = obj.transform(scaled_data)
        return x

    def sparse_pca(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = SparsePCA(n_components=n_components,random_state=0)
        obj.fit(data)
        x = obj.transform(data)
        return x

    def truncated_svd(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = TruncatedSVD(n_components=n_components,random_state=0)
        obj.fit(scaled_data)
        x = obj.transform(scaled_data)
        return x

    #metrics
    def calc_acc(self,model,X_test,Y_test):
        y_pred = model.predict(X_test) 
        return accuracy_score(Y_test,y_pred),y_pred


    '''def date_col(self,df,col):df
        #Transform string to date
        df['date'] = pd.to_datetime(df.date, format="%d-%m-%Y")

        #Extracting Year
        df['year'] = df['date'].dt.year

        #Extracting Month
        df['month'] = df['date'].dt.month

        #Extracting passed years since the date
        df['passed_years'] = date.today().year - df['date'].dt.year

        #Extracting passed months since the date
        df['passed_months'] = (date.today().year - df['date'].dt.year) * 12 + date.today().month - df['date'].dt.month

        #Extracting the weekday name of the date
        df['day_name'] = df['date'].dt.day_name()'''










        

    