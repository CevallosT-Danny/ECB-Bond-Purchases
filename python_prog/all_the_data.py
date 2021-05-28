#!/usr/bin/env python
# coding: utf-8

# In[207]:


import pandas as pd
import glob
import dateutil.parser as dparser


# In[142]:


# import cbonds list
cbonds_path = r'/Users/gabbyvinco/Desktop/HISTORICAL_BONDS.csv'
cbonds = pd.read_csv(cbonds_path, index_col=None, header=0)


# In[143]:


q2_2017_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2017/Q2'
q3_2017_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2017/Q3'
q4_2017_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2017/Q4'
q1_2018_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2018/Q1'
q2_2018_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2018/Q2'
q3_2018_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2018/Q3'
q4_2018_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2018/Q4'
q1_2019_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2019/Q1'
q2_2019_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2019/Q2'
q3_2019_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2019/Q3'
q4_2019_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2019/Q4'
q1_2020_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2020/Q1'
q2_2020_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2020/Q2'
q3_2020_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2020/Q3'
q4_2020_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2020/Q4'
q1_2021_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2021/Q1'
q2_2021_path = r'/Users/gabbyvinco/Desktop/historical_lists_cspp/2021/Q2'

list_of_paths = [q2_2017_path, q3_2017_path, q4_2017_path, 
                 q1_2018_path, q2_2018_path, q3_2018_path, 
                 q4_2018_path, q1_2019_path, q2_2019_path, 
                 q3_2019_path, q4_2019_path, q1_2020_path, 
                 q2_2020_path, q3_2020_path, q4_2020_path, 
                 q1_2021_path, q2_2021_path]


# In[144]:


complete_list = []


# In[145]:


for path in list_of_paths:
    all_files = glob.glob(path + "/*.csv")
    complete_list.append(all_files)


# In[146]:


complete_list[1][1][-12:]


# In[147]:


list_of_df_names = ["q2_2017_df", "q3_2017_df", "q4_2017_df", "q1_2018_df", 
 "q2_2018_df", "q3_2018_df", "q4_2018_df", "q1_2019_df", 
 "q2_2019_df", "q3_2019_df", "q4_2019_df", "q1_2020_df", 
 "q2_2020_df", "q3_2020_df", "q4_2020_df", "q1_2021_df", 
 "q2_2021_df"]


# In[148]:


all_historical_data = []
df_names = []


# In[149]:


for i in range(len(list_of_df_names)):
    link_length = len(complete_list[i])
    df_name = list_of_df_names[i]
    for l in range(link_length):
        individual_csv = complete_list[i][l]
        csv_name = complete_list[i][l][-12:]
        csv = pd.read_csv(individual_csv, encoding='latin-1')
        all_historical_data.append(csv)
        df_names.append(csv_name)


# In[150]:


all_historical_data[1]


# In[151]:


for n in range(len(df_names)):
    week_gathered = df_names[n]
    all_historical_data[n]["historical_date"] = week_gathered


# In[152]:


# all_historical_data


# In[153]:


all_historical_copies = pd.concat(all_historical_data)


# In[154]:


all_historical_copies


# In[155]:


all_historical_copies["ISIN_Code"] = all_historical_copies["ISIN_CODE"].astype(str) + all_historical_copies["ISIN"].astype(str)
all_historical_copies["Issuer_Name"] = all_historical_copies["ISSUER_NAME_"].astype(str) + all_historical_copies["ISSUER_NAME"].astype(str) + all_historical_copies["ISSUER"].astype(str)
all_historical_copies["Maturity_Date"] = all_historical_copies["MATURITY_DATE_"].astype(str) + all_historical_copies["MATURITY_DATE"].astype(str) + all_historical_copies["MATURITY DATE"].astype(str)
all_historical_copies["Coupon_Rate"] = all_historical_copies["COUPON_RATE_"].astype(str) + all_historical_copies["COUPON_RATE_*"].astype(str) + all_historical_copies["COUPON_RATE"].astype(str) + all_historical_copies["COUPON RATE"].astype(str)


# In[98]:


all_historical_copies


# In[156]:


# list(all_historical_copies.columns.values)


# In[157]:


all_historical_copies = all_historical_copies.drop(columns=['ISIN_CODE',
                                                            'ISSUER_NAME_',
                                                            'MATURITY_DATE_',
                                                            'COUPON_RATE_',
                                                            'COUPON_RATE_*',
                                                            'ISSUER_NAME',
                                                            'MATURITY_DATE',
                                                            'COUPON_RATE',
                                                            'Unnamed: 5',
                                                            'Unnamed: 6',
                                                            'Unnamed: 0',
                                                            'ISIN',
                                                            'ISSUER',
                                                            'MATURITY DATE',
                                                            'COUPON RATE'])


# In[158]:


all_historical_copies = all_historical_copies.reset_index(drop = True)


# In[159]:


all_historical_copies.loc[0]["historical_date"]


# In[160]:


all_historical_copies.head()


# In[161]:


# split the dataframes into 


# In[163]:


for row in range(len(all_historical_copies)):
    all_historical_copies.iloc[row, 1] = all_historical_copies.loc[row]["historical_date"].replace('.csv', '')
    all_historical_copies.iloc[row, 2] = all_historical_copies.loc[row]["ISIN_Code"].replace('nan', '')
    all_historical_copies.iloc[row, 3] = all_historical_copies.loc[row]["Issuer_Name"].replace('nan', '')
#     all_historical_copies.iloc[row, 3] = all_historical_copies.loc[row]["Issuer_Name"].replace('nannan', '')
    all_historical_copies.iloc[row, 4] = all_historical_copies.loc[row]["Maturity_Date"].replace('nan', '')
#     all_historical_copies.iloc[row, 4] = all_historical_copies.loc[row]["Maturity_Date"].replace('nannan', '')
    all_historical_copies.iloc[row, 5] = all_historical_copies.loc[row]["Coupon_Rate"].replace('nan', '')
#     all_historical_copies.iloc[row, 5] = all_historical_copies.loc[row]["Coupon_Rate"].replace('nannan', '')
#     all_historical_copies.iloc[row, 5] = all_historical_copies.loc[row]["Coupon_Rate"].replace('nannannan', '')


# In[164]:


all_historical_copies.head()


# In[165]:


len(all_historical_copies)


# In[166]:


all_isin_codes = all_historical_copies["ISIN_Code"].tolist()


# In[167]:


# all_isin_codes


# In[168]:


cbonds_isin_list = cbonds["ISIN"].tolist()


# In[169]:


# set(all_isin_codes).symmetric_difference(set(cbonds_isin_list))


# In[ ]:





# In[ ]:





# In[170]:


included_isin = all_historical_copies.loc[all_historical_copies['ISIN_Code'].isin(cbonds_isin_list)]


# In[172]:


included_isin = included_isin.sort_values(by=['ISIN_Code'])


# In[173]:


included_isin


# In[174]:


unique = included_isin.ISIN_Code.unique()


# In[175]:


len(unique)


# In[183]:


unique[:5]


# In[178]:


value = pd.DataFrame()


# In[179]:


for item in unique:
    info = cbonds.loc[cbonds["ISIN"] == item]
    value = value.append(info, ignore_index = True)


# In[185]:


value = value.rename(columns={"ISIN": "ISIN_Code"})


# In[186]:


value.head()


# In[187]:


included_isin


# In[188]:


combo = included_isin.merge(value, how='left', on='ISIN_Code')


# In[191]:


combo.info()


# In[192]:


print(combo.shape)


# In[195]:


combo = combo.sort_values(by=['ISIN_Code', "historical_date"])


# In[197]:


combo.head(10)


# In[ ]:


combo=All_of_the_data


# In[190]:


# save the unique_improved to csv file
combo.to_csv (r'/Users/gabbyvinco/Desktop/All_of_the_data.csv', index = False, header=True)


# In[ ]:




