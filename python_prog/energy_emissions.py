#!/usr/bin/env python
# coding: utf-8

# # Emissions in the Green Economy

# In[1]:


# Graphs for main notebook: fig1, fig6 and scatter2.


# In[36]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.cluster import KMeans
from io import BytesIO
import requests
from PIL import Image


# ## Data preprocessing and Exploratory analysis

# Data is prepocessed to evaluate the emissions related to production and consumption.

# ### Production and Consumption

# In[4]:


bal = pd.read_csv("https://www.dropbox.com/s/pz1i1qb62q5dzzg/nrg_bal_s.tsv?dl=1", header=0,delimiter='\t')
baldf = pd.concat([bal.iloc[:,0].str.split(',', expand=True), bal.iloc[:,1:] ], axis=1)
baldf.columns.values[list(range(0,4))] = ['item/sector','technology','unit','country']
baldf


# In[16]:


def get_bal(item, tech, unit):
    
    # choose item/sector in 0
    # [PPRD] Primary production
    # [FC_IND_E] Final consumption - industry sector - energy use
    # [FC_TRA_E] Final consumption - transport sector - energy use

    # choose which technology in 1
    #  [E7000] Electricity #use for consumption, not for production
    #  [RA000] Renewables and biofuels #use for consumption or production
    #  [TOTAL] Total

    # see all labels
    # https://ec.europa.eu/eurostat/databrowser/view/nrg_bal_s/default/table?lang=en
    
    out = baldf[(baldf.iloc[:,0]==item) & (baldf.iloc[:,1]==tech) & (baldf.iloc[:,2]==unit)]
    return out
    
prod_green = get_bal('PPRD','RA000','GWH')
prod_tot = get_bal('PPRD','TOTAL','GWH')
tra_green = get_bal('FC_TRA_ROAD_E','E7000','GWH')
tra_tot = get_bal('FC_TRA_ROAD_E','TOTAL','GWH')
prod_green.head(n=3)


# In[17]:


def clean_bal(df):
    mid = pd.melt(df, id_vars='country', value_vars=df.columns[4:-10],value_name='gwh')
    out = mid.replace(": ","")
    out['gwh'] = pd.to_numeric(out['gwh'])
    return out
    
greenp = clean_bal(prod_green)
totp = clean_bal(prod_tot)
greent = clean_bal(tra_green)
tott = clean_bal(tra_tot)
greenp


# In[18]:


totp


# In[ ]:





# In[19]:


def nation(df,countrylist):
    # filter countries

    out= df[df['country'].isin(countrylist)]
    return out

countrylist = ['AT','BE','BG', 'CY','CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IS','IT',
                   'LU','LT', 'LV', 'NL', 'NO', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK', 'UK']
greenp = nation(greenp,countrylist)
totp = nation(totp,countrylist)
greent = nation(greent,countrylist)
tott = nation(tott,countrylist)


# In[20]:


def div(a,b):
    r = np.divide(a, b, where=a.dtypes.ne(object)).combine_first(a)
    r.columns = ['country','year','ratio']
    return r

ratio = div(greenp,totp)
ratiot = div(greent,tott)
ratio


# In[42]:


def to_wide(df):
    out = pd.pivot(df,index='year', columns='country', values='ratio')
    return out

wide = to_wide(ratio)
widet = to_wide(ratiot)
wide


# ##### Production

# The results from the correspondence analysis showed countries like Portugal and Italy were highly related to green bonds and bonds for the power industry. 
# 
# In this figure we see why. The ratio of clean production is plotted over time to make fair comparisons across countries with different capacities. Portugal has had an almost complete ratio of clean energy production for the past two decades. Italy has a higher and slightly steeper evolution of the ratio compared to other countries. These countries, in which clean production is the norm or very mature, are good assets for banks like the ECB to invest through their bond purchasing programs. More investment in them can trigger a snowball effect, by which a better maintained infrastructure attracts even more investment. Another reason for their advantage is their climate and location.

# In[29]:


sns.set(rc={'figure.figsize':(11,7)})

wide = wide[['PL','FR','DE','UK','NO','PT','IT']]
fig1 = wide.plot();
fig1.set_xlabel("Year");
fig1.set_ylabel("Ratio Clean Prod");
plt.savefig('prod_plot.png')
plt.close()


# In[38]:


#plot saved image
prod_plot0 = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/prod_plot.png')
prod_plot = Image.open(BytesIO(prod_plot0.content))
prod_plot


# ##### Consumption

# It is also interesting to see how the ratio of electric road energy consumption evolves over time for the transport sector.

# In[43]:


widet = widet[['PL','FR','DE','UK','NO']]
fig2 = widet.plot()
fig2.set_xlabel("Year")
fig2.set_ylabel("Ratio Road Electric  Cons")
plt.savefig('consp_plot1.png')
plt.close()

widet = widet[['PL','FR','DE','UK']]
fig2b = widet.plot();
fig2b.set_xlabel("Year")
fig2b.set_ylabel("Ratio Electric Road Cons")
plt.savefig('consp_plot2.png')
plt.close()


# In[45]:


#plot saved image
consp_plot10 = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/consp_plot1.png')
consp_plot1 = Image.open(BytesIO(consp_plot10.content))
consp_plot1


# In[46]:


consp_plot20 = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/consp_plot2.png')
consp_plot2 = Image.open(BytesIO(consp_plot20.content))
consp_plot2


# ### Emissions

# The aim of this section is to evaluate what are the emissions associated to the production and consumption explored before.

# In[50]:

emiss = pd.read_csv("https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Datasets/env_air_gge.tsv", header=0,delimiter='\t')
emissdf = pd.concat([emiss.iloc[:,0].str.split(',', expand=True), emiss.iloc[:,1:] ], axis=1)
emissdf.columns.values[list(range(0,4))] = ['unit', 'chem','sector','country']
emissdf


# In[51]:


def get_em(unit, chem, item):
    
    # choose which sector in 2
    #  [TOTX4_MEMO] Total (excluding LULUCF and memo items)
    #  [CRF1A3] Fuel combustion in transport
    #  [CRF2] Industrial processes and product use #not as general as the industry sector from before
    #  [CRF3] Agriculture
    #  [CRF4] Land use, land use change, and forestry (LULUCF)

    # see all labels
    # https://ec.europa.eu/eurostat/databrowser/view/env_air_gge/default/table?lang=en
    
    out = emissdf[(emissdf.iloc[:,0]==unit) & (emissdf.iloc[:,1]==chem) & (emissdf.iloc[:,2]==item)]
    return out
    
emiss_prod = get_em('MIO_T','GHG','CRF1A1A')
emiss_tra = get_em('MIO_T','GHG','CRF1A3B')


# In[52]:


def clean_em(df):
    out = pd.melt(df, id_vars='country', value_vars=df.columns[4:-15],value_name='ghg')
    out['ghg'] = pd.to_numeric(out['ghg'])
    return out

emissp = clean_em(emiss_prod)
emisst = clean_em(emiss_tra)
emissp


# In[53]:


# filter by country
emissp = nation(emissp,countrylist)
emisst = nation(emisst,countrylist)


# In[54]:


def comb(df_em,df_bal_r, df_bal_gwh,countrylist):
    df_bal_r = df_bal_r[len(countrylist):]
    df_bal_gwh = df_bal_gwh[len(countrylist):]
    out = df_em
    out['ratio'] =pd.to_numeric(df_bal_r['ratio']).to_numpy()
    out['gwh'] =pd.to_numeric(df_bal_gwh['gwh']).to_numpy()
    return out

emiss_ft2 = comb(emisst,ratiot,greent,countrylist)
emiss_fp2 = comb(emissp,ratio,greenp,countrylist)


# ###### Consumption

# In this case it is not very clear how emissions evolve with the ratio or the amount of energy consumed by electric road vehicles in the transport sector.

# In[55]:


fig3 = sns.lmplot(x='gwh', y='ghg', data=emiss_ft2[emiss_ft2['country'].isin(['PL','FR','DE','NO','UK','PT','IT'])], hue='country', fit_reg=False);
fig3.set(xlabel='Road Electricity consumed in GWH', ylabel='GHG in MIOT')
plt.savefig('emtype_comsum1.png')
plt.close()

fig4 = sns.lmplot(x='ratio', y='ghg', data=emiss_ft2[emiss_ft2['country'].isin(['PL','FR','DE','NO','UK','PT','IT'])], hue='country', fit_reg=False);
fig4.set(xlabel='Ratio Road Electric  Cons', ylabel='GHG in MIOT')
plt.savefig('emtype_comsum2.png')
plt.close()


# In[56]:


emtype_comsum10 = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/emtype_comsum1.png')
emtype_comsum1 = Image.open(BytesIO(emtype_comsum10.content))
emtype_comsum1


# In[57]:


emtype_comsum20 = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/emtype_comsum2.png')
emtype_comsum2 = Image.open(BytesIO(emtype_comsum20.content))
emtype_comsum2


# ##### Production

# In this case, when the values of production are plotted against the emissions there are clear correlations, for most countries. The second figure is of special interest. It shows how the emissions of GHG decrease as the clean ratio of production grows. A linear slope for each country could be useful to rank them.

# In[58]:


fig5 = sns.lmplot(x='gwh', y='ghg', data=emiss_fp2[emiss_fp2['country'].isin(['PL','FR','DE','NO','UK','PT','IT'])], hue='country', fit_reg=False);
fig5.set(xlabel='Electricity consumed in GWH', ylabel='GHG in MIOT')
plt.savefig('emtype_prod1.png')
plt.close()

fig6 = sns.lmplot(x='ratio', y='ghg', data=emiss_fp2[emiss_fp2['country'].isin(['PL','FR','DE','NO','UK','PT','IT'])], hue='country', fit_reg=False);
fig6.set(xlabel='Ratio Clean Prod', ylabel='GHG in MIOT');
plt.savefig('emtype_prod2.png')
plt.close()


# In[59]:


emtype_prod10 = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/emtype_prod1.png')
emtype_prod1 = Image.open(BytesIO(emtype_prod10.content))
emtype_prod1


# In[60]:


emtype_prod20 = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/emtype_prod2.png')
emtype_prod2 = Image.open(BytesIO(emtype_prod20.content))
emtype_prod2


# ## Classification of countries

# Following up the previous idea, the linear slope coefficient for every country is obtained. The first ranked country is the UK.

# In[63]:


def ols_res(df, xcol,  ycol):
    x =sm.add_constant(df[xcol])
    return sm.OLS(df[ycol], x).fit().params

sol = emiss_fp2.groupby('country').apply(ols_res, xcol='ratio', ycol='ghg')
sol.columns = ['const','slope']
em_Reg=sol.sort_values(by='slope')
em_Reg.T


# Not only the slope is important, but also their mean level of GHG emissions. Since a country could be very slowly reducing their emissions, but could be already in low levels.

# In[64]:


# create the mean ghg per country for the K-means algorithm
media = emiss_fp2.groupby('country').mean()
sol2 = pd.concat([sol['slope'], media], axis=1)
sol2['country'] = sol2.index
sol2.sort_values(by='slope');


# For every country, there is a slope of emission and a mean value of emissions available. It can be interesting to see which countries have similar characteristics of these two values by grouping them together using a K-means algorithm with 4 clusters.

# In[70]:


X = sol2.drop(['ratio','gwh','country'], axis=1)
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
sol2['label'] = kmeans.labels_
sol2
sol3 = sol2[sol2['country'].isin(['PL','FR','DE','NO','UK','PT','NL','BE','DK','IT','IS','IE','AT'])]
grouped_country = sol3
grouped_country

# The results for all countries are presented below. Some conclusions can be made:
# 
# - Countries are grouped mostly by their slope, except for **Germany and Poland**. These countries have been labelled 2 because of their high value of mean emissions. This suggests that further changes (substituting coal power plants for others) have to be done in their production plan if they aim to be part of the countries labelled 3. These countries, for a similar slope, have a much lower mean emission value. 
# 
# - The **UK** is labelled 1, as the country with the best reduction of emissions. While at the same time, maintaining a reasonable mean emission value.
# 
# - Countries labelled 0, have a low value of the mean emissions but not a great reduction value. The case of **Netherlands** stands out as having the biggest mean emission value of its group.
# 
# - **Norway** is the only country with a positive slope, although it has one of the lowest mean emission values.
# 
# 

# In[71]:


sns.set(rc={'figure.figsize':(11,7)})

scatter = sns.scatterplot(x='slope', y='ghg', data=sol2[sol2['label']==0], hue='country', size= 'label', sizes={0:25,1:75,2:200,3:400}, legend = 'full') 
scatter = sns.scatterplot(x='slope', y='ghg', data=sol2[sol2['label']==1], hue='country', size= 'label', sizes={0:25,1:75,2:200,3:400}, legend = 'brief') 
scatter = sns.scatterplot(x='slope', y='ghg', data=sol2[sol2['label']==2], hue='country', size= 'label', sizes={0:25,1:75,2:200,3:400}, legend = 'brief') 
scatter = sns.scatterplot(x='slope', y='ghg', data=sol2[sol2['label']==3], hue='country', size= 'label', sizes={0:25,1:75,2:200,3:400}, legend = 'brief') 

scatter.legend(bbox_to_anchor= (1.04,1.25),fontsize=9);
scatter.set(xlabel='Slope of Emiss vs Ratio', ylabel='GHG in MIOT')
plt.savefig('country_kmeans1.png')
plt.close()

sns.set(rc={'figure.figsize':(11,7)})

scatter2 = sns.scatterplot(x='slope', y='ghg', data=sol3[sol3['label']==0], hue='country', size= 'label', sizes={0:25,1:75,2:200,3:400}, legend = 'full') 
scatter2 = sns.scatterplot(x='slope', y='ghg', data=sol3[sol3['label']==1], hue='country', size= 'label', sizes={0:25,1:75,2:200,3:400}, legend = 'brief') 
scatter2 = sns.scatterplot(x='slope', y='ghg', data=sol3[sol3['label']==2], hue='country', size= 'label', sizes={0:25,1:75,2:200,3:400}, legend = 'brief') 
scatter2 = sns.scatterplot(x='slope', y='ghg', data=sol3[sol3['label']==3], hue='country', size= 'label', sizes={0:25,1:75,2:200,3:400}, legend = 'brief') 

scatter2.legend(bbox_to_anchor= (1.15,1.0),fontsize=9);
scatter2.set(xlabel='Slope of Emiss vs Ratio', ylabel='GHG in MIOT')
plt.savefig('country_kmeans2.png')
plt.close()


# In[68]:


country_kmeans10 = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/country_kmeans1.png')
country_kmeans1 = Image.open(BytesIO(country_kmeans10.content))
country_kmeans1


# Here are the same results presented for a limited number of countries.

# In[69]:


country_kmeans20 = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/country_kmeans2.png')
country_kmeans2 = Image.open(BytesIO(country_kmeans20.content))
country_kmeans2

