#!/usr/bin/env python
# coding: utf-8

# # Trend in bonds investment:

# ## Data processing

# In[47]:

from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import warnings
import itertools
import numpy as np
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# In[2]:


bonds = pd.read_csv("https://www.dropbox.com/s/70ig02z77ranw6x/All_of_the_data.csv?dl=0",header=0)
bonds.head(n=3)


# In[3]:


bonds.dtypes


# In[4]:


##Subsetting columns
bonds=bonds[["industry","country","bond_types","ISIN_Code","total_bond_amount","coupon_rate","nominal_amount","maturity_date",'historical_date']]
bonds.shape


# In[5]:


#Deletion of null bond types
bonds=bonds[~pd.isnull(bonds["bond_types"])]
bonds.shape


# In[6]:


#Grouping bond types
auxgreen=["Green" in  aux for aux in bonds["bond_types"]]
bonds.iloc[auxgreen,2]="Green bonds"

auxzero=["Zero" in  aux for aux in bonds["bond_types"]]
bonds.iloc[auxzero,2]="Zero-coupon bonds"

auxforeign=["Foreign" in  aux for aux in bonds["bond_types"]]
bonds.iloc[auxforeign,2]="Foreign bonds"

auxfloating=["Floating" in  aux for aux in bonds["bond_types"]]
bonds.iloc[auxfloating,2]="Floating rate"

bonds["bond_types"][bonds["bond_types"].isin(["Senior Secured", "Floating rate", "Foreign bonds","Securitization"])]="Others"


# In[7]:


bonds["bond_types"].value_counts()


# In[8]:


#Groupipng industries
bonds["industry"][bonds["industry"].isin(["Pulp, paper and wood industries", "Ferrous metals", "Mining industry","Non-ferrous metals"])]="Other sectors"
bonds["industry"].value_counts()


# In[9]:


#Convert integer to date format
bonds['historical_date'] = pd.to_datetime(bonds['historical_date'], format='%Y%m%d')
bonds.sort_values(by=['historical_date'], inplace=True)


bonds['week']=bonds['historical_date'].dt.week.astype(int)
bonds['year']=bonds['historical_date'].dt.year.astype(int)


# In[10]:


#Assigning a week number to each obs
bonds['extra_weeks_2017'] = np.where(bonds['year']==2017, -25, 0)
bonds['extra_weeks_2018'] = np.where(bonds['year']>=2018, 27, 0)
bonds['extra_weeks_2019'] = np.where(bonds['year']>=2019, 52, 0)
bonds['extra_weeks_2020'] = np.where(bonds['year']>=2020, 52, 0)
bonds['extra_weeks_2021'] = np.where(bonds['year']>=2021, 51, 0)


sum_column = bonds['week']+ bonds['extra_weeks_2017']+ bonds['extra_weeks_2018']+ bonds['extra_weeks_2019']+ bonds['extra_weeks_2020']+ bonds['extra_weeks_2021']
bonds["n_week"] = sum_column


# In[11]:


##Transform nominal_amount into int
bonds[['amount_str', 'EUR']] = bonds['nominal_amount'].str.split(' ', 1, expand=True)
bonds['nom_amount']=bonds['amount_str'].str.replace(',','')
bonds['nom_amount']=bonds['nom_amount'].astype(int)


# ## Investment by bond type

# In[29]:


##Total investment by week and bond type
cum_bonds=bonds.groupby(['n_week','bond_types'], as_index=False).agg(total_invest=('nom_amount', sum))
cum_bonds.T


# In[28]:


#Total investment per week
sum_week=cum_bonds.groupby('n_week', as_index=False).sum()
sum_week.T


# In[30]:


total_per_week=[]
for i in range(0, 201, 1):
    repeated=  itertools.repeat(sum_week['total_invest'][i], 4)
    total_per_week.extend(repeated)
cum_bonds['weekly_total']=total_per_week
#Percentage of weekly total invest
cum_bonds['week_percent']=cum_bonds['total_invest']/cum_bonds['weekly_total']


# In[31]:


#Curves per bond type
others=cum_bonds[cum_bonds['bond_types']=='Others']
green=cum_bonds[cum_bonds['bond_types']=='Green bonds']
unsecured=cum_bonds[cum_bonds['bond_types']=='Senior Unsecured']
zero=cum_bonds[cum_bonds['bond_types']=='Zero-coupon bonds']


# In[69]:


plt.figure(figsize=(20, 8))
plt.plot(others['n_week'], others['week_percent'], 'b-', label = 'others')
plt.plot(green['n_week'], green['week_percent'], 'g-', label = 'green')
plt.plot(unsecured['n_week'], unsecured['week_percent'], '-', label = 'unsecured')
plt.plot(zero['n_week'], zero['week_percent'], 'y-', label = 'zero')
plt.xlabel('week'); plt.ylabel('Euros'); plt.title('Investment progression')
plt.legend();
plt.savefig('inv_plot.png')
plt.close()


# In[70]:


#plot saved image
response = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/inv_plot.png')
inv_prog = Image.open(BytesIO(response.content))


# In[72]:


##Total investment by week and bond type
cum_industry=bonds.groupby(['n_week','industry'], as_index=False).agg(total_invest=('nom_amount', sum))


# In[73]:


#Total investment per week
sum_week_industry=cum_industry.groupby('n_week', as_index=False).sum()
cum_industry.T


# In[74]:


total_per_week=[]
for i in range(0, 201, 1):
    repeated=  itertools.repeat(sum_week_industry['total_invest'][i], 16)
    total_per_week.extend(repeated)
cum_industry['weekly_total']=total_per_week
#Percentage of weekly total invest
cum_industry['week_percent']=cum_industry['total_invest']/cum_industry['weekly_total']


# ## Investment by industry

# In[75]:


chemical_petrochemical_industry=cum_industry[cum_industry['industry']=='Chemical and petrochemical industry']                                   
Construction_development=cum_industry[cum_industry['industry']=='Construction and development']                                    
Food_industry =cum_industry[cum_industry['industry']=='Food industry']                                     
Oil_gas=cum_industry[cum_industry['industry']=='Oil and gas']                                   
Engineering_industry=cum_industry[cum_industry['industry']=='Engineering industry']                                    
Information_High_Technologies=cum_industry[cum_industry['industry']=='Information and High Technologies']                                    
Power =cum_industry[cum_industry['industry']=='Power']                                     
Media_Entertainment =cum_industry[cum_industry['industry']=='Media and Entertainment']                                    
Communication=cum_industry[cum_industry['industry']=='Communication']                                    
Public_utilities =cum_industry[cum_industry['industry']=='Public utilities']                                     
Financial_institutions =cum_industry[cum_industry['industry']=='Financial institutions']                                     
Transportation=cum_industry[cum_industry['industry']=='Transportation']                                    
Other_sectors =cum_industry[cum_industry['industry']=='Other sectors']                                     
Healthсare =cum_industry[cum_industry['industry']=='Healthсare']                                     
Light_industry=cum_industry[cum_industry['industry']=='Light industry']                                  
Trade_retail=cum_industry[cum_industry['industry']=='Trade and retail']  


# In[76]:


plt.figure(figsize=(20, 8))

plt.plot(chemical_petrochemical_industry['n_week'],chemical_petrochemical_industry['week_percent'], 'd-', label = 'chemical_petrochemical_industry')                       
plt.plot(Construction_development['n_week'],Construction_development['week_percent'], 'b-', label = 'Construction_development')          
plt.plot(Food_industry['n_week'],Food_industry['week_percent'], 'c-', label = 'Food_industry')
plt.plot(Oil_gas['n_week'],Oil_gas['week_percent'],'d-', label = 'Oil_gas')          
plt.plot(Engineering_industry['n_week'],Engineering_industry['week_percent'], 'y-', label = 'Engineering_industry')                                                          
plt.plot(Information_High_Technologies['n_week'],Information_High_Technologies['week_percent'], 'g-', label = 'Information_High_Technologies')                                          
plt.plot(Power['n_week'],Power['week_percent'], 'r-', label = 'Power')                          
plt.plot(Media_Entertainment['n_week'],Media_Entertainment['week_percent'], 'h-', label = 'Media_Entertainment')                                                               
plt.plot(Communication['n_week'],Communication['week_percent'], 'p-', label = 'Communication')                                         
plt.plot(Public_utilities['n_week'],Public_utilities['week_percent'], 'r-', label = 'Public_utilities')                                                            
plt.plot(Financial_institutions['n_week'],Financial_institutions['week_percent'], 'p-', label = 'Financial_institutions')                                         
plt.plot(Transportation['n_week'],Transportation['week_percent'], 'w-', label = 'Transportation')                                                          
plt.plot(Other_sectors['n_week'],Other_sectors['week_percent'], 'm-', label = 'Other_sectors')                                                            
plt.plot(Healthсare['n_week'],Healthсare['week_percent'], 'h-', label = 'Healthсare')                                                            
plt.plot(Light_industry['n_week'],Light_industry['week_percent'], 'o-', label = 'Light_industry')                                                        
plt.plot(Trade_retail['n_week'],Trade_retail['week_percent'], 'p-', label = 'Trade_retail') 

plt.xlabel('week'); plt.ylabel('Euros'); plt.title('Investment progression')

plt.legend();
plt.savefig('inv_plot_ind.png')
plt.close()

# In[79]:


#plot saved image
response = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/inv_plot_ind.png')
inv_prog_ind = Image.open(BytesIO(response.content))

