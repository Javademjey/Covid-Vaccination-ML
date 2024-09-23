import streamlit as st
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler ,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
from xgboost import XGBRegressor ,plot_importance



x = pd.read_csv("country_vaccinations.csv")

countries_list = x['country'].unique()
vaccines_list = x['vaccines'].unique()

x.drop(columns=['iso_code' ,'daily_vaccinations' ,
                'daily_vaccinations_per_million' ,
                'source_name' ,'source_website'] ,inplace=True)

org_df = x.copy()

SI = SimpleImputer(missing_values=np.nan ,strategy='mean')
x .iloc[:,2:-1]= SI.fit_transform(x.iloc[:,2:-1])

x['day'] = x['date'].str[8:]
x['month'] = x['date'].str[5:7]
x['year'] = x['date'].str[:4]
x.drop(['date'] ,axis=1 ,inplace=True)

LE = LabelEncoder()
x['country'] = LE.fit_transform(x['country'])
x['vaccines'] = LE.fit_transform(x['vaccines'])

x = x.drop(['total_vaccinations'] ,axis=1)
col = x.columns
col=col[:-1]

X = x.drop(['total_vaccinations'] ,axis=1).values
y = x['total_vaccinations'].values

X[:,[-1,-2,-3]] = X[:,[-1,-2,-3]].astype(float)

X = StandardScaler().fit_transform(X)  

x_train ,x_test ,y_train ,y_test = train_test_split(X ,y ,test_size=0.2 ,random_state=4)

XGBR = XGBRegressor(n_estimators=1000 ,learning_rate=0.11 ,max_depth=6  ,reg_alpha= 14)
XGBR.fit(x_train,y_train)



# --------------------------------------------------------------------------------------------------------




st.sidebar.header("Parameters")

def user_input_features():
    country = st.sidebar.selectbox('Name Of Country' ,options=countries_list)
    vaccines = st.sidebar.selectbox('Name Of Vaccines' ,options=vaccines_list)
    date = st.sidebar.date_input("Date", dt.date(2020, 7, 6))
    total_vaccinations = st.sidebar.slider('Total vaccinations' ,0.0 ,1426347000.0 ,12004435.0)
    people_vaccinated = st.sidebar.slider('Teople Taccinated',0.0 ,622000000.0 ,5704550.0 )
    people_fully_vaccinated = st.sidebar.slider('People Fully Vaccinated', 1.0, 223299000.0, 3293973.0)
    daily_vaccinations_raw = st.sidebar.slider('Daily Vaccinations Raw', 0.0, 24741000.0, 227366.0)
    total_vaccinations_per_hundred = st.sidebar.slider('Total Vaccinations Per Hundred', 0.0, 232.0, 29.0)
    people_vaccinated_per_hundred = st.sidebar.slider('People Vaccinated Per Hundred', 0.0, 116.0, 19.0)
    people_fully_vaccinated_per_hundred = st.sidebar.slider('People Fully Vaccinated Per Hundred' ,0.0 ,115.0 ,115.0)
    
    year = date.year
    month = date.month
    day = date.day
    
    data1 = {'country': country,
            'vaccines': vaccines
            }
    data2 = {'daily_vaccinations_raw': daily_vaccinations_raw,
            'total_vaccinations_per_hundred': total_vaccinations_per_hundred,
            'people_vaccinated_per_hundred': people_vaccinated_per_hundred,
            'people_fully_vaccinated_per_hundred': people_fully_vaccinated_per_hundred,
            'people_vaccinated': people_vaccinated,
            'people_fully_vaccinated': people_fully_vaccinated,
            'year': year,
            'month': month,
            'day': day
            }
    
    features1 = pd.DataFrame(data1, index=[0] ,dtype='category')
    features2= pd.DataFrame(data2, index=[0])
    user_df = pd.concat([features1 ,features2] ,axis=1)
    user_df_encoded = user_df.copy()

    countries_dict = dict(zip(countries_list ,range(218)))
    vaccines_dict = dict(zip(vaccines_list ,range(53)))

    user_df_encoded = user_df.replace({'vaccines': vaccines_dict,
                                      'country': countries_dict})  

    return user_df ,user_df_encoded

user_df ,user_df_encoded = user_input_features()
st.subheader('User Input parameters')
st.write(user_df)

st.subheader('Encoded parameters')
st.write(user_df_encoded)

st.sidebar.write("")
st.sidebar.write("")

prediction = XGBR.predict(user_df_encoded)
st.subheader('Prediction')
st.write(pd.DataFrame({'total_vaccinations':prediction } ,index=['0']))


def DownloadDataset(df ,org_df):
    
    def convert_df(df):
        return df.to_csv().encode("utf-8")

    data = st.sidebar.radio("Which data do you want to download ?" ,
                    options=["Original Data" ,"Encoded Data"])
    if data == "Original Data":
        csv = convert_df(org_df)
    else:
        csv = convert_df(df)

    st.sidebar.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="Dataset.csv",
        mime="text/csv",
        )

DownloadDataset(x ,org_df)


importances = XGBR.feature_importances_
importances_seri = pd.DataFrame(importances[:-1], index=col).sort_values(0 ,ascending=False)
dict_impor = importances_seri.to_dict()
dict_impor = dict_impor[0]
feature_name = list(dict_impor.keys())
feature_score = list(dict_impor.values())

fig, ax = plt.subplots(figsize=(7,5))
plot_importance(XGBR, max_num_features=len(col), ax=ax)
plt.yticks(range(len(feature_name[::-1])) ,feature_name[::-1])
plt.show() 

