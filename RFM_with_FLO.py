###############################################################
# Customer Segmentation Using RFM with FLO Dataset
###############################################################

###############################################################
# 1. Business Problem
###############################################################
#FLO company wants to segment its customers and determine marketing strategies according to these segments.
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

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_=pd.read_csv("flo_data_20k.csv")
df=df_.copy()

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

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

df.groupby("order_channel").agg({"master_id":"count",
                                 "order_num_total":"sum",
                                 "customer_value_total":"sum"})

df.sort_values("customer_value_total", ascending=False)[:10] #the top 10 customers with the highest revenue.

df.sort_values("order_num_total", ascending=False)[:10] # the top 10 customers with the most orders.

###############################################################
# 4.Calculating RFM Metrics
###############################################################

df.head()
df["last_order_date"].max() # 2021-05-30

today_date = dt.datetime(2021,6,1)


rfm = df.groupby('master_id').agg({'last_order_date': lambda date: (today_date - date.max()).days,
                                     'order_num_total': lambda num: num,
                                     'customer_value_total': lambda a: a})

rfm.columns = ['recency', 'frequency', 'monetary']
rfm.head()

###############################################################
# 5.Calculating RFM Scores
###############################################################

rfm["recency_score"]=pd.qcut(rfm["recency"],5,labels=[5,4,3,2,1])

rfm["monetary_score"]=pd.qcut(rfm["monetary"],5,labels=[1,2,3,4,5])

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.head()

###############################################################
# 6. Creating & Analysing RFM Segments
###############################################################

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm.head()

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["min","mean","max","count"])

new_df = pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index
new_df.to_csv("new_customers.csv")
rfm.to_csv("rfm.csv")

###############################################################
#7. Functionalize the data preparation process
###############################################################

def create_rfm(dataframe,csv=False,csv_name="csv"):

    # Data Understanding
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Columns #####################")
    print(dataframe.columns)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe().T)
    print("##################### Types #####################")
    print(dataframe.dtypes)

    # Data Preparation
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]

    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    dataframe.groupby("order_channel").agg({"master_id": "count",
                                     "order_num_total": "sum",
                                     "customer_value_total": "sum"})

    dataframe.sort_values("customer_value_total", ascending=False)[:10]  # the top 10 customers with the highest revenue.

    dataframe.sort_values("order_num_total", ascending=False)[:10]  # the top 10 customers with the most orders.

    # Calculating RFM Metric
    dataframe["last_order_date"].max()  # 2021-05-30

    today_date = dt.datetime(2021, 6, 1)

    rfm = dataframe.groupby('master_id').agg({'last_order_date': lambda date: (today_date - date.max()).days,
                                       'order_num_total': lambda num: num,
                                       'customer_value_total': lambda a: a})

    rfm.columns = ['recency', 'frequency', 'monetary']

    # Calculating RFM Scores
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))

    # Creating & Analysing RFM Segments
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]

    if csv:
        rfm.to_csv(csv_name)

    return rfm

df=df_.copy()
rfm_new = create_rfm(df, csv=True,csv_name="k")
rfm_new.head()

#Case1:Customers will be contacted specifically with champions, loyal_customers and shoppers from the women category.
# Save the id numbers of these customers in the csv file.

a=pd.DataFrame(rfm_new[(rfm_new["segment"]=="champions") |(rfm_new["segment"]=="loyal_customers")].index)
b=df[df["interested_in_categories_12"].str.contains("KADIN",na=False)]["master_id"]
id_list=pd.merge(a, b)
id_list.to_csv("ıd_list_csv")

#Case2:Up to 40% discount is planned for Men's and Children's products.
# Long-standing but past good customers interested in categories related to this discount non-shopping and new customers are specifically targeted.
# Enter the ids of the customers in the appropriate profile into the csv file discount_target_customer_ids.csv. Save it as

k=pd.DataFrame(rfm_new[(rfm_new["segment"]=="cant_loose") |(rfm_new["segment"]=="hibernating")|(rfm_new["segment"]=="new_customers")].index)
m=df[(df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK"))]["master_id"]
target_customer=pd.merge(k,m)
target_customer.to_csv("target_customer_csv")
