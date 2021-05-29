#!/usr/bin/env python
# coding: utf-8

# In[1]:



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
#%pip install mca
import mca
from PIL import Image
from io import BytesIO
import requests
from sklearn import linear_model
import seaborn as sns
sns.set()

import itertools


# In[ ]:


pd.set_option('display.max_columns', 100000)
pd.set_option('display.max_rows', 100000)


# # Loading of Data

# In[2]:


df=pd.read_csv("https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Datasets/HISTORICAL_BONDS.csv",header=0)
df.head(n=3)


# In[3]:


df_sm=df[["industry","country","bond_types","ISIN","total_bond_amount","coupon_rate","nominal_amount","maturity_date","year_created"]]
df_sm.shape


# There are 2,083 bonds

# In[4]:


df_sm.dtypes


# In[5]:


df_sm.describe(exclude=np.number)


# In[6]:


1206/2083


# Only 58% of the bonds have assigned a "bond type". Since this characteristic is of importance to detect if the bank has helped or not the green economy, we will only use in the analysis bonds with a non-missing bond type.

# In[7]:


#Imputation of data for bonds
df_sm=df_sm[~pd.isnull(df_sm["bond_types"])]
df_sm.shape


# Number of bonds by industry

# In[8]:


df_sm["industry"].value_counts()


# We group industries with low observations and add them into the "other sectors" group. The purpose of this is to not draw assumptions from them in the correspondence analysis that are only based in a couple of observations

# In[9]:


df_sm["industry"][df_sm["industry"].isin(["Pulp, paper and wood industries", "Ferrous metals", "Mining industry","Non-ferrous metals"])]="Other sectors"


# In[10]:


df_sm["industry"].value_counts()


# We group bonds too

# In[11]:


df_sm["bond_types"].value_counts()


# In[12]:


#Grouping of bond types
auxgreen=["Green" in  aux for aux in df_sm["bond_types"]]
df_sm.iloc[auxgreen,2]="Green bonds"

auxzero=["Zero" in  aux for aux in df_sm["bond_types"]]
df_sm.iloc[auxzero,2]="Zero-coupon bonds"

auxforeign=["Foreign" in  aux for aux in df_sm["bond_types"]]
df_sm.iloc[auxforeign,2]="Foreign bonds"

auxfloating=["Floating" in  aux for aux in df_sm["bond_types"]]
df_sm.iloc[auxfloating,2]="Floating rate"

df_sm["bond_types"][df_sm["bond_types"].isin(["Senior Secured", "Floating rate", "Foreign bonds","Securitization"])]="Others"


# In[13]:


df_sm["bond_types"].value_counts()


# We visualize the frequency of each of the bond type by country, for descriptive purposes

# In[14]:


crosst=pd.crosstab(df_sm["country"],df_sm["bond_types"])
crosst


# Number of bonds per country. We will group countries which are similar and which have a very small number of bonds.

# In[15]:


df_sm["country"].value_counts() 


# In[16]:


df_sm["country"][df_sm["country"].isin(["USA","Hong Kong", "South Africa", "Bermuda"])]="Non_EU"
df_sm["country"][df_sm["country"].isin(["Sweden", "Denmark"])]="Sweden_Denmark"
df_sm["country"][df_sm["country"].isin(["Romania", "Slovakia","Czech Republic"])]="CZ_SK_RO"
df_sm["country"][df_sm["country"].isin(["Lithuania", "Estonia"])]="LT_EE"
df_sm["country"][df_sm["country"].isin(["Belgium","Luxembourg"])]="BE_LU"


# In[17]:


df_sm["country"].value_counts()


# # Descriptive Analysis: Correspondence Analysis
# 
# We analyse the relationships between industry, country and type of a bond

# In[ ]:





# In[18]:


df_dummies=pd.get_dummies(df_sm[["industry","country","bond_types"]], prefix=None, prefix_sep='_')


# In[19]:


df_dummies.shape
df_dummies.head(3)


# In[20]:


df_dummies


# In[21]:


mca_ind = mca.MCA(df_dummies, benzecri=True)
mca_ind


# In[22]:


len(mca_ind.L) #One factor for level
inertias=mca_ind.L  #Eigenvalues/principal inertias of each of the factors
inertias


# Factors of each observation

# In[23]:


fs, cos, cont = 'Factor score','Squared cosines', 'Contributions x 1000'
table3 = pd.DataFrame(columns=df_dummies.index, index=pd.MultiIndex
                      .from_product([[fs, cos, cont], range(1, 3)]))

table3.loc[fs,    :] = mca_ind.fs_r(N=2).T
table3.loc[cos,   :] = mca_ind.cos_r(N=2).T
table3.loc[cont,  :] = mca_ind.cont_r(N=2).T * 1000

np.round(table3.astype(float), 2)


# Factors of each dummy variable

# In[24]:


table4 = pd.DataFrame(columns=df_dummies.columns, index=pd.MultiIndex
                      .from_product([[fs, cos, cont], range(1, 3)]))
table4.loc[fs,  :] = mca_ind.fs_c(N=2).T
table4.loc[cos, :] = mca_ind.cos_c(N=2).T
table4.loc[cont,:] = mca_ind.cont_c(N=2).T * 1000

table4.shape
np.round(table4.astype(float), 2)


# In[25]:


Factors=table4.iloc[:2,:].T
Factors.columns=["Factor_1","Factor_2"]
Factors.T


# In[26]:

# Plot of the first 2 factors of the correspondence analysis, for each of the dummy variables

# In[27]:


labels=['Chem/Petro','Commun.', 'Construction','    Engineering', 'Financial','   Food', 'Healthсare',
       'Technology', '    Light','Entertainment', '        Oil/Gas',' Others', 'Power', 'Public utilities',
       'Trade', 'Trans.','AT', 'BE-LU', '    CZ-SK-RO','FI', 'FR', 'DE','IE', 'IT', 'LT-EE', 'NL', '  Non-EU',
       'PT', 'ES', '    SE-DK','CH', '            UK','Green bonds', '      Others','Sr. Unsec.', 'Zero-coup.']


# In[28]:


CA_plot, ax=plt.subplots(figsize=(20, 12))
ax.scatter(Factors["Factor_1"], Factors["Factor_2"], s=[200]*16+[200]*16+[600]*4, color=["darkred"]*16+["green"]*16+["navy"]*4) # c=Df_PCA_Biplot["Compound"], vmin=-0.6, vmax=0.6,cmap = matplotlib.cm.get_cmap('jet_r')

ax.set_ylabel("Factor 2",size=40)
ax.set_xlabel("Factor 1",size=40)
ax.set_title("Relationship between Bond Type, \n Country and Industry",size=60)

for i, txt in enumerate(Factors.index):
    if i>=32:
        ax.annotate(labels[i], (Factors["Factor_1"][i]-0.2, Factors["Factor_2"][i]+0.07),size=23)
    else:
        ax.annotate(labels[i], (Factors["Factor_1"][i], Factors["Factor_2"][i]),size=20)
        
Bond_patch = mpatches.Patch(color='navy', label='Bond Types')
Country_patch = mpatches.Patch(color='green', label='Countries')
Industry_patch = mpatches.Patch(color='darkred', label='Industries')
ax.legend(handles=[Bond_patch,Country_patch,Industry_patch], fontsize=30,loc="lower left")
plt.savefig('Cor_An.png')
plt.close()


# In[29]:


#plot saved image
Cor_An0 = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/Cor_An.png')
Cor_An = Image.open(BytesIO(Cor_An0.content))


# Portugal, Lithuania-Estonia, Italy and Ireland, are the countries most related to green bonds.
# 
# The Power industry is extremely related to these bonds too.

# In[ ]:





# # Data Pre-processing

# ## Data Cleaning

# Here we will:
# 
# * Change the format of numeric data
# * Add a rate to floating rate bonds. We will add the mean of the rates
# * Change the format of date columns and add year columns
# 

# Change format of numeric data

# In[30]:


#df_sm["total_bond_amount"]=df_sm["total_bond_amount"].str.replace('EUR','').str.replace(',','').astype(float)


# In[31]:


df_sm["nominal_amount"]=df_sm["nominal_amount"].str.replace('EUR','').str.replace(',','').astype(float)


# In[32]:


df_sm["coupon_rate"]=df_sm["coupon_rate"].str.replace('%','').str.replace('FRN','nan').astype(float)


# Add a rate to floating rate bonds (mean)

# In[33]:


df_sm["coupon_rate"][pd.isnull(df_sm["coupon_rate"])]=df_sm["coupon_rate"].mean()


# Change format of date columns and add year columns

# In[34]:


df_sm['maturity_date']=pd.to_datetime(df_sm['maturity_date'])


# In[35]:


df_sm['maturity_year']=df_sm['maturity_date'].dt.year


# In[36]:


df_sm['year_created']=df_sm['year_created'].astype(int)


# In[37]:


df_sm[['maturity_year','year_created']].describe()


# In[ ]:





# We will now obtain a "surplus" for the company for each of the bond. For that, we obtain the value of all the interest and nominal value that the company has to pay to the bank, at the year that the bond was purchased. We then rest this amount to the total bond amount. In this way we will obtain the "surplus" of the deal for the company's side. 
# 
# After that, since different bond could have different issuing dates, we will take all the surplus values to the 2021 year. In this way we can make the surplus of the bonds comparable.
# 
# For making this process, we assign to floating rate bonds the average rate, and we considered a fix inlation rate of the one present in the euro area at the time this notebook was done, which is 1.6%.

# ## Annuities and present value functions

# In[38]:


def startvalue_zerobond(yr_start,yr_end,inflation,facevalue):
    diffyears=yr_end-yr_start
    totalvalue_atstart=facevalue/((1+inflation)**diffyears)
    return totalvalue_atstart

def startvalue_interest(yr_start,yr_end,inflation,interest,bondcost):
    diffyears=yr_end-yr_start
    totalvalue_atstart=((bondcost*interest)/(1+inflation))*((1-(1+inflation)**(-diffyears))/(1-(1+inflation)**(-1)))
    return totalvalue_atstart

def startvalue_completebond(yr_start,yr_end,inflation,facevalue,interest,bondcost):
    zerobond_atstart=startvalue_zerobond(yr_start,yr_end,inflation,facevalue)
    interest_atstart=startvalue_interest(yr_start,yr_end,inflation,interest,bondcost)
    all_value_atstart=interest_atstart+zerobond_atstart
    return all_value_atstart

def fromstart_topresent(yr_start,yr_present,value,inflation):
    diffyears=yr_present-yr_start
    presentvalue=value*(1+inflation)**diffyears
    return presentvalue


# In[39]:


df_sm.head(3)


# ## Creation of surplus at time 2021

# Getting the start value of contributions to the company for the bank. We suppose an inflation of 1.6%, and that the bank bought the bond at the nominal value.

# In[40]:


df_sm['Payment_atstartvalue']=startvalue_completebond(df_sm['year_created'],df_sm['maturity_year'],0.016,df_sm['nominal_amount'],
                        df_sm['coupon_rate']/100,df_sm['nominal_amount'])


# Getting the surplus at time that the bond was purchased

# In[41]:


df_sm['surplus_company']=df_sm['nominal_amount']-df_sm['Payment_atstartvalue']


# Getting the surplus value at time 2021

# In[42]:


df_sm['surplus_company_21']= fromstart_topresent(df_sm['year_created'],2021, df_sm['surplus_company']   ,0.016 )           


# In[43]:


df_sm['surplus_company_21'].describe()


# In[44]:


df_sm.sort_values(by='surplus_company_21')


# # Distribution of green bonds compared to the rest of them

# Distribution of the surplus at 2021 

# In[45]:


df_sm['surplus_company_21'].describe()


# In[46]:


plt.subplots(figsize=(20, 12))
sns.distplot(df_sm['surplus_company_21'], hist = True, kde = True,bins=30,
                 kde_kws = {'shade': True, 'linewidth': 3})#, label = bond_types)
plt.close()


# Distribution by bond type

# In[47]:


plt.subplots(figsize=(20, 12))
sns.violinplot(x ="bond_types", y ="surplus_company_21", data = df_sm)# ,hue ="region", style ="event"
plt.close()


# Small signs that the bank could be supporting the green bonds so far. The bond that seems to be supported the most is the Zero-coupon bond.

# In[48]:


df_sm["Ind_Green"]="No"
df_sm["Ind_Green"][df_sm["bond_types"]=="Green bonds"]="Yes"
df_sm["Ind_Green"].value_counts()


# Distribution of surplus by industry and bond type

# In[49]:


plt.subplots(figsize=(10, 30))
sns.violinplot(y ="industry", x ="surplus_company_21", data = df_sm,width=1,dodge=True, style ="event",hue ="Ind_Green",
               split=True)# 
plt.legend(fontsize=20,title="Green \n Bond",title_fontsize=23, loc="upper left")
plt.xticks(size = 15)
plt.yticks(size = 20)
plt.xlabel("Surplus at 2021",size=25)
plt.ylabel("")
plt.title("Distribution Surplus \n by Industry at 2021", size=25);
plt.savefig('Sur_2021.png')
plt.close()


# In[50]:


#plot saved image
Sur_20210 = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/Sur_2021.png')
Sur_2021 = Image.open(BytesIO(Sur_20210.content))


# Surplus in engineering seems bigger for green bonds than the rest, but the sample of green bonds (5) is too small in those industries for making strong conclussions.

# In[51]:


ind_green=pd.crosstab(df_sm["Ind_Green"],df_sm["industry"]).T
ind_green.T


# In[ ]:





# # Lasso

# We already know that the number of green bonds is very small compared to the total number of bonds bought by the bank. 
# 
# In order to detect if the bank is helping the companies that buy green bonds in a significant way, we will do a penalized regression technique, particularly Lasso regression, on the surplus at time 2021 in millions, with the variables already analyzed as explanatory variables (industry, country, and type of bond).
# 
# If we were to make a model using the dummy variables we have analyzed so far as covariates, we may go into over fitting issues due to the high amount of covariates, so the best for us will be to implement some kind of feature selection before fitting a model.
# 
# Hence, we decided to implement a Lasso model because of its nature to shrink the less important coefficients to zero. In this way, if after fitting the model, the coefficient associated to green bonds is positive, we can say that the bank is supporting the green economy.
# 
# We did not standardize our covariates since all of them are dummy variables. Standardizing will not bring any value, but it will make interpretation of the coefficients more difficult.
# 
# For hyperparameter tunning as well as model selection, we will make use of 10 fold cross validation

# In[52]:


id_bond=df_sm["ISIN"]
X=pd.get_dummies(df_sm[["industry","country","bond_types"]], prefix=None, prefix_sep='_')
y=df_sm["surplus_company_21"]


# ## Impact of the alpha value to the Lasso coefficients

# We set our possible alpha values from 0 to 15, because after 15 all coefficients were zero

# In[53]:


alpha = np.arange(0,250,0.5)
#alpha


# Matrix that will store for each alpha, the coefficients values

# In[54]:


weights = np.zeros((len(alpha),len(X.columns)),dtype=float) #matrix of dimensions #alphas x #dummievars
weights.shape


# In[55]:


for i in np.arange(0,len(alpha)):
    lassomodelaux = linear_model.Lasso(fit_intercept=True, alpha=alpha[i],copy_X=True,normalize=False)
    lassomodelaux.fit(X,y)
    weights[i,:]= lassomodelaux.coef_  #matrix defined above starts to be filled up
    


# We plot the evolution of the coefficients as the alpha increases

# In[56]:


ab = itertools.chain(["cadetblue"]*16+['mediumseagreen']*16+["darkred"]+["lightcoral"]*3)
colorlist=list(ab)


# In[57]:


ab=itertools.chain([3]*20+[7]*2+[3]*4+[3]*6+[7]+[3]*1+[7]*2)
sizelist=list(ab)


# In[58]:


labelsaux=['Chem/Petro','Commun.', 'Construction','Engineering', 'Financial','Food', 'Healthсare',
       'Technology', 'Light','Entertainment', 'Oil/Gas',' Others', 'Power', 'Public utilities',
       'Trade', 'Trans.','AT', 'BE-LU', 'CZ-SK-RO','FI', 'FR', 'DE','IE', 'IT', 'LT-EE', 'NL', 'Non-EU',
       'PT', 'ES', 'SE-DK','CH', 'UK','Green bonds', 'Others','Sr. Unsec.', 'Zero-coup.']


# In[59]:


plt.figure(figsize=(20,15))    
for i in np.arange(0,len(X.columns)):
    plt.semilogx(alpha[:],weights[:,i],label=labelsaux[i], linewidth=sizelist[i],
                 color=colorlist[i]) #semilog to make right values closer

plt.xlabel('Alpha Value',size=30)
plt.ylabel('Coefficient',size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(loc="lower right", ncol=5,fontsize=20);
plt.savefig('Las_alfa.png')
plt.close()


# In[60]:


Las_alfa0 = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/Las_alfa.png')
Las_alfa = Image.open(BytesIO(Las_alfa0.content))


# The green bonds seem to shrink to zero slower than most of the variables, but still is not part of the group variables that lasted the most.
# 
# The coefficients that survive the most are the ones related to Zero-coupon and Senior Unsecured bonds, as well as Germany and France

# ## Cross validation for hyperparameter tuning and model selection

# In[61]:


alpha[:10]


# In[62]:


# use cross validation using 10 Fold cross validation
Lasso=linear_model.LassoCV(alphas=alpha,fit_intercept=True,normalize=False,cv=10)
Lasso.fit(X,y)


# In[63]:


#print('Optimal Value for Alpha:{v:0.4f}'.format(v=Lasso.alpha_))
#print('\n')
#print('Lasso coefficients: ')
#print(Lasso.coef_)
#print('\n')
#print('Lasso intercept:')
#print(Lasso.intercept_)


# The alpha that decreases the loss function the most is 30

# In[68]:


Lasso_Coeffs, ax = plt.subplots(1,1,sharex=True,figsize=(10,15))
y_pos = np.arange(len(X.columns))

#Lasso Regression
hedge_Lasso = Lasso.coef_
ax.barh(y=y_pos, width=hedge_Lasso, align='center', alpha=0.9,
        color=["cadetblue"]*16+['mediumseagreen']*16+["darkred"]+["lightcoral"]*3)
ax.set_yticks(y_pos)
plt.yticks(size=15)
ax.set_yticklabels(X.columns)
ax.set_xlabel('Coefficient Value', size=20)
ax.set_title('Coefficients for the Lasso Regression Model', size=25);
plt.savefig('Lasso_p.png')
plt.close()


# In[65]:


Lasso_p0 = requests.get('https://github.com/Dansieg91/ECB-Bond-Purchases/raw/main/Plots/Lasso_p.png')
Lasso_p = Image.open(BytesIO(Lasso_p0.content))
Lasso_p


# In[ ]:





# In[66]:


ResultingDf=pd.DataFrame(data={"Dummy_variable":X.columns,"Coefficient":np.round(Lasso.coef_,2),"N_bonds":X.sum()})
ResultingDf=ResultingDf.reset_index()[["Dummy_variable","Coefficient","N_bonds"]]
Las_cv=ResultingDf.sort_values(by="Coefficient", ascending=False)
Las_cv


# # Conclussions
# 
# <b>Conclussion: Yes, the European Central bank has been supporting the green economy when purchasing corporate
# bonds, although it is not the main priority for the bank to support.</b>
# 
# Normally the bonds that are supported the most are the ones whose company is located in "France" and "Germany", as well as the "Zero-coupon" bonds.
# 
# The big coefficient for the Zero coupon bonds can be explain more than the decision of the bank to support those types of bonds, because of the way we created our outcome variable. Since we were not able to find the exact price at which the bank bought the bond, we used the face value as the price. Due to the nature of Zero-coupon bonds of not giving any interest, this bonds are usually bought at a smaller price than the face value. Thats why simply by construction of our model, the surplus for Zero-coupon bonds will be high.
# 
# But the Zero-coupon bonds only account for the 7% percent of the analyzed bonds. Regarding the other types of bonds (93% of the total), all three coefficients were non-zero, but the one associated to green-bonds is positive (beta=127) whereas the other two are negative and big (beta= -750 and -524). Since the difference between the coefficients of green-bonds and the other two is very big, we can conclude that the ECB is supporting more the green-bonds than the other two types of bonds.
# 
# 
# Furthermore, we saw in the correspondence analysis that the industry "Power" was very related to green bonds. In this model, the coefficient "Power" although small, is positive (beta=83), so these corroborates again the conclussion that the European Central Bank is supporting the green economy.
# 
# We also see that the fact of the green bond coefficient being zero could not be entirely due to the small number of green bonds, since we see that 5 of the other 11 non-zero coefficients are related to variables whose number of bonds belonging to is lower than the ones for green bonds.

# # Study limitations
# 
# Although helpful, this analysis is not perfect since we made some assumptions beforehand. Some of the assumptions we made were:
# 
# * <b>Fixed inflation rate</b>
# 
# For calculating the present value of the contributions of the company to the ECB, we considered a fixed inflation rate.
# 
# * <b>Cost bond equal to the face value</b>
# 
# Given the fact that we were not able to find the exact cost at which the ECB bought the bonds, we were forced to use the face value of a bond as its price. This will not be realistic to happen for Zero-coupon bonds.
# 
# 
# 
# Additionally, for calculating the present value of the interest and face value, we did not considered the exact date at which the bonds were issued or matured, we only the year of those dates.
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




