###############################################################
# Customer Lifetime Value Prediction Using RFM with FLO Dataset
###############################################################

###############################################################
# 1. Business Problem
###############################################################
#FLO wants to set a roadmap for sales and marketing activities.
#In order for the company to make a medium-long-term plan, it is necessary to estimate the potential value that existing customers will provide to the company in the future.
#The dataset consists of the information obtained from the past shopping behaviors of the customers who made
# their last shopping from Flo as OmniChannel (both online and offline shopping) in the years 2020-2021.

# master_id: Unique customer number
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
# first_order_date : Date of the customer's first purchase
# last_order_date : Customer's last purchase date
# last_order_date_online : The date of the last purchase made by the customer on the online platform
# last_order_date_offline : Last shopping date made by the customer on the offline platform
# order_num_total_ever_online : The total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : Total number of purchases made by the customer offline
# customer_value_total_ever_offline : Total fee paid by the customer for offline purchases
# customer_value_total_ever_online : The total fee paid by the customer for their online shopping
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months

###############################################################
# 2. Data Understanding
###############################################################
import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width',1000)

df_=pd.read_csv("flo_data_20k.csv")
df=df_.copy()

df.head()

def check_df(dataframe, head=10):
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Columns #####################")
    print(dataframe.columns)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe().T)
    print("##################### Types #####################")
    print(dataframe.dtypes)

check_df(df, head=10)

###############################################################
# 3.Data Preparation
###############################################################
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

k=[col for col in df.columns if "ever" in col]

for col in k:
    print(col, check_outlier(df, col))
    replace_with_thresholds(df, col)

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df["first_order_date"]=df["first_order_date"].astype("datetime64[ns]")
df["last_order_date"]=df["last_order_date"].astype("datetime64[ns]")
df["last_order_date_online"]=df["last_order_date_online"].astype("datetime64[ns]")
df["last_order_date_offline"]=df["last_order_date_offline"].astype("datetime64[ns]")

###############################################################
# 4.Creating the CLTV Data Structure
###############################################################
df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)

cltv_df1 = df.groupby("master_id").agg(
    { "first_order_date": lambda x: x,
      "last_order_date": lambda y: y})

cltv_df2 = df.groupby("master_id").agg(
    { "first_order_date":  lambda k: (analysis_date - k),
      'order_num_total': lambda num: num,
      'customer_value_total': lambda TotalPrice: TotalPrice})

cltv_df1["recency_cltv_weekly"]=cltv_df1["last_order_date"]-cltv_df1["first_order_date"]

cltv_df1.drop("first_order_date", axis=1, inplace=True)
cltv_df1.drop("last_order_date", axis=1, inplace=True)

cltv_df = cltv_df1.merge(cltv_df2, on="master_id")

cltv_df.columns = ['recency_cltv_weekly', 'T_weekly', 'frequency', 'monetary_cltv_avg']

cltv_df["monetary_cltv_avg"] = cltv_df["monetary_cltv_avg"] / cltv_df["frequency"]

cltv_df["recency_cltv_weekly"] = (cltv_df["recency_cltv_weekly"]/ 7).astype('timedelta64[D]')

cltv_df["T_weekly"] = (cltv_df["T_weekly"] / 7).astype('timedelta64[D]')

cltv_df["frequency"]=cltv_df["frequency"].astype("int")

cltv_df.head()

###############################################################
# 5. Expected Number of Transaction with BG-NBD Model
###############################################################
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

bgf.predict(1,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])

cltv_df["exp_sales_3_month"]=bgf.predict(3*4,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])

cltv_df["exp_sales_6_month"]=bgf.predict(6*4,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])

cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]

cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]

###############################################################
# 6. Expected Average Profit with Gamma-Gamma Model
###############################################################
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'],
        cltv_df['monetary_cltv_avg'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])
cltv_df.head()

###############################################################
# 7. Calculation of CLTV with BG-NBD and Gamma-Gamma Model
###############################################################
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 month
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

cltv_df.sort_values("cltv",ascending=False)[:20]

###############################################################
#8.Creating Segments by CLTV
###############################################################

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_df.head()

cltv_df.groupby("cltv_segment").agg(["max","mean","count"])


