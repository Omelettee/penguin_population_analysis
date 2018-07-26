# for the purpose of statistics, we applied a time series model 
# of ARMA for prediction the pupulation of pengusin. 

from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
import random
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LinearRegression

penguin_original = pd.read_csv('data_original/penguin_population.csv')
penguin = penguin_original[['species','year','month','season_starting','population','ccamlr_region','count_type']]
penguin = penguin[penguin.population !='None']
penguin.population = penguin.population.astype('int')
#penguin.year = penguin.year.astype('int')

penguin_group = penguin.groupby(['species','year'])[['population']].mean()
penguin_group = penguin_group.reset_index()

gentoo= penguin_group[penguin_group.species == 'gentoo penguin']
chinstrap = penguin_group[penguin_group.species == 'chinstrap penguin']
emperor = penguin_group[penguin_group.species == 'emperor penguin']
adelie = penguin_group[penguin_group.species == 'adelie penguin']
global_original = pd.read_csv('data_original/global_annual_temp.csv')
global_temp = global_original[['Year','DJF','JJA']]
global_temp.columns = ['year','global_summer','global_winter']


sh_original = pd.read_csv('data_original/sh_annual_temp.csv')
sh_temp = sh_original[['Year','DJF','JJA']]
sh_temp.columns = ['year','sh_summer','sh_winter']

zone_original = pd.read_csv('data_original/zone_temp.csv')
zone_temp = zone_original[['Year','90S-64S']]
zone_temp.columns = ['year','Ant_annual']
zone_temp = zone_temp[zone_temp['Ant_annual'] != 'None']
zone_temp.Ant_annual = zone_temp.Ant_annual.astype('float')


ice_area = pd.read_csv('data_original/ice_area.csv')
ice_area.columns = ['year','month','Weddell','Indian','Pacific','Ross','Bell_Amu','Ant']
ice_area.month = ice_area.month.astype('int')
ice_area_annual = ice_area.groupby(['year'])[['Weddell','Indian','Pacific','Ross','Bell_Amu','Ant']].mean()
ice_area_annual = ice_area_annual.reset_index()
ice_area_annual.columns =['year','Weddell_ice_area_annual','Indian_ice_area_annual','Pacific_ice_area_annual','Ross_ice_area_annual','Bell_Amu_ice_area_annual','Ant_ice_area_annual']
ice_area_summer = ice_area[(ice_area['month']>=1) & ice_area['month']<=3]
ice_area_summer = ice_area_summer.groupby(['year'])[['Weddell','Indian','Pacific','Ross','Bell_Amu','Ant']].mean()
ice_area_summer = ice_area_summer.reset_index()
ice_area_summer.columns =['year','Weddell_ice_area_summer','Indian_ice_area_summer','Pacific_ice_area_summer','Ross_ice_area_summer','Bell_Amu_ice_area_summer','Ant_ice_area_summer']
ice_area_winter = ice_area[(ice_area['month']>=6) & ice_area['month']<=8]
ice_area_winter = ice_area_winter.groupby(['year'])[['Weddell','Indian','Pacific','Ross','Bell_Amu','Ant']].mean()
ice_area_winter = ice_area_winter.reset_index()
ice_area_winter.columns =['year','Weddell_ice_area_winter','Indian_ice_area_winter','Pacific_ice_area_winter','Ross_ice_area_winter','Bell_Amu_ice_area_winter','Ant_ice_area_winter']


ice_extent = pd.read_csv('data_original/ice_extent.csv')
ice_extent.columns = ['year','month','Weddell','Indian','Pacific','Ross','Bell_Amu','Ant']
ice_extent.month = ice_extent.month.astype('int')
ice_extent_annual = ice_extent.groupby(['year'])[['Weddell','Indian','Pacific','Ross','Bell_Amu','Ant']].mean()
ice_extent_annual = ice_extent_annual.reset_index()
ice_extent_annual.columns =['year','Weddell_ice_extent_annual','Indian_ice_extent_annual','Pacific_ice_extent_annual','Ross_ice_extent_annual','Bell_Amu_ice_extent_annual','Ant_ice_extent_annual']
ice_extent_summer = ice_extent[(ice_extent['month']>=1) & ice_extent['month']<=3]
ice_extent_summer = ice_extent_summer.groupby(['year'])[['Weddell','Indian','Pacific','Ross','Bell_Amu','Ant']].mean()
ice_extent_summer = ice_extent_summer.reset_index()
ice_extent_summer.columns =['year','Weddell_ice_extent_summer','Indian_ice_extent_summer','Pacific_ice_extent_summer','Ross_ice_extent_summer','Bell_Amu_ice_extent_summer','Ant_ice_extent_summer']
ice_extent_winter = ice_extent[(ice_extent['month']>=6) & ice_extent['month']<=8]
ice_extent_winter = ice_extent_winter.groupby(['year'])[['Weddell','Indian','Pacific','Ross','Bell_Amu','Ant']].mean()
ice_extent_winter = ice_extent_winter.reset_index()
ice_extent_winter.columns =['year','Weddell_ice_extent_winter','Indian_ice_extent_winter','Pacific_ice_extent_winter','Ross_ice_extent_winter','Bell_Amu_ice_extent_winter','Ant_ice_extent_winter']

krill_catch_origianl = pd.read_csv('data_original/krill_catch.csv')
krill_catch = krill_catch_origianl[['year','month','Krill_Green_Weight/tons']]
krill_catch.columns = ['year','month','weight']
krill_catch.month = krill_catch.month.astype('int')
krill_catch_annual = krill_catch.groupby(['year'])[['weight']].sum()
krill_catch_annual = krill_catch_annual.reset_index()
krill_catch_annual.columns = ['year','annual_catch']
krill_catch_summer = krill_catch[(krill_catch['month']>=1) & krill_catch['month']<=3]
krill_catch_summer = krill_catch_summer.groupby(['year'])[['weight']].sum()
krill_catch_summer = krill_catch_summer.reset_index()
krill_catch_summer.columns = ['year','summer_catch']
krill_catch_winter = krill_catch[(ice_extent['month']>=6) & krill_catch['month']<=8]
krill_catch_winter = krill_catch_winter.groupby(['year'])[['weight']].sum()
krill_catch_winter = krill_catch_winter.reset_index()
krill_catch_winter.columns = ['year','winter_catch']


penguin_list = [[adelie,'adelie'],[gentoo,'gentoo'],[chinstrap,'chinstrap'],[emperor,'emperor']]
fit_list = [global_temp,zone_temp,ice_area_annual,ice_extent_annual,krill_catch_annual]
#adelie_join = reduce(lambda left,right : pd.merge(left,right,on='year'),adelie_list)
#adelie.set_index('year').join(global_temp.set_index('year'),lsuffix='_left', how='inner')

for t in penguin_list:
    for i in fit_list:
        t[0] = pd.merge(t[0],i,on = ['year','year'])
#        print t[0].shape
    feature_list = t[0].columns
    feature_value = []
    for i in range (3,t[0].shape[1]):
        x = [float(m) for m in t[0].iloc[:,2]]
        y = [float(m) for m in t[0].iloc[:,i]]
 #       print pearsonr(x,y)
        feature_value.append((pearsonr(x,y)[0]**2,pearsonr(x,y)[0],feature_list[i]))

    feature_value.sort(key=lambda x:x[0], reverse = True)
    print feature_value[:3]
    x_axis = range(1,11)
    y_axis = [i[0] for i in feature_value[:10]]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.barh(x_axis,y_axis,alpha=0.5, label =t[1])
    ax.set_yticks(range(1, 11))
    ax.set_yticklabels([i[2] for i in feature_value[:10]])
    plt.xlabel("Pearson Correlation")
    plt.legend(loc=1)
    plt.gcf().subplots_adjust(left=0.35)
    plt.show()
    
#s=pd.DataFrame(columns = ['species','year','population'])     
#ks=pd.DataFrame(columns = ['species','year','population'])     
s=adelie 
for i in [gentoo,chinstrap,emperor]:
    s=pd.merge(s,i,on=['year','year'])
    #print s.shape
    #print s
s.columns = ['a','year','adelie','b','gentoo','c','chinstrap','d','emperor']    
penguin_joined = s[['year','adelie','gentoo','chinstrap','emperor']]   

#1. adelie
adelie_joined = [krill_catch_annual,ice_extent_annual,ice_area_annual]
gentoo_joined = [global_temp,krill_catch_annual,ice_area_annual]
emperor_joined = [ice_area_annual,ice_extent_annual,krill_catch_annual]
chinstrap_joined = [ice_extent_annual,ice_area_annual]

penguin_joined = [adelie,gentoo,emperor,chinstrap]
feature_joined = [adelie_joined,gentoo_joined,emperor_joined,chinstrap_joined]
features1 = [['population','annual_catch','Ant_ice_area_annual','Ant_ice_extent_annual'],
            ['population','global_winter','annual_catch','Ross_ice_area_annual'],
            ['population','Ross_ice_area_annual','Ross_ice_extent_annual','annual_catch'],
            ['population','Weddell_ice_extent_annual','Pacific_ice_area_annual','Weddell_ice_area_annual']]

features2 = [['year','population','annual_catch','Ant_ice_area_annual','Ant_ice_extent_annual'],
            ['year','population','global_winter','annual_catch','Ross_ice_area_annual'],
            ['year','population','Ross_ice_area_annual','Ross_ice_extent_annual','annual_catch'],
            ['year','population','Weddell_ice_extent_annual','Pacific_ice_area_annual','Weddell_ice_area_annual']]
penguin_names=['adelie','gentoo','emperor','chinstrap']
for i in range(len(penguin_joined)):
    s = penguin_joined[i]
    for j in range(len(feature_joined[i])):
        s=pd.merge(s,feature_joined[i][j], on=['year','year'])
    #print s.shape    
    #print features[i]
    temp=s[features2[i]]
#    print temp.shape
    temp_list = [temp[features2[i][k]].values.tolist() for k in range(len(features2[i]))]
    #print temp_list
    plt.subplots(figsize=(9,9))
    ax1 = plt.subplot(411) 
    plt.plot(temp_list[0],temp_list[1],'o-',label=penguin_names[i])
    plt.legend(loc=1)
    plt.setp(ax1.get_xticklabels(),visible=False)
    plt.setp(ax1.get_yticklabels(),visible=False)
    plt.ylabel(features2[i][1])
    plt.tick_params(axis='both',which='both',bottom='off',left='off')
    ax2 = plt.subplot(412,sharex = ax1)
    plt.plot(temp_list[0],temp_list[2],'o-')
    plt.setp(ax2.get_xticklabels(),visible=False)
    plt.setp(ax2.get_yticklabels(),visible=False)
    plt.ylabel(features2[i][2])
    plt.tick_params(axis='both',which='both',bottom='off',left='off')
    ax3 = plt.subplot(413,sharex = ax1)
    plt.plot(temp_list[0],temp_list[3],'o-')
    plt.setp(ax3.get_xticklabels(),visible=False)
    plt.setp(ax3.get_yticklabels(),visible=False)
    plt.ylabel(features2[i][3])
    plt.tick_params(axis='both',which='both',bottom='off',left='off')
    ax4 = plt.subplot(414,sharex = ax1)
    plt.setp(ax4.get_yticklabels(),visible=False)
    plt.plot(temp_list[0],temp_list[4],'o-')
    plt.setp(ax4.get_xticklabels(),fontsize=10)
    plt.ylabel(features2[i][4])
    plt.xlabel(features2[i][0])
    plt.tick_params(axis='y',which='both',bottom='off')
    plt.savefig('%s.png' %penguin_names[i])


for i in range(len(penguin_joined)):
    s = penguin_joined[i]
    for j in range(len(feature_joined[i])):
        s=pd.merge(s,feature_joined[i][j], on=['year','year'])
    #print s.shape    
    #print features[i]
    temp=s[features1[i]]
#    print temp.shape
    temp_list = [temp[features1[i][k]].values.tolist() for k in range(len(features1[i]))]
    temp = temp.values 
#    print type(temp[0][0])
    X = temp[:,1:]
    Y = temp[:,0]
#    print X.shape
    #x_train,x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.15)
    linear = LinearRegression()
    linear.fit(X,Y)
    linear_test_predict = [k for k in linear.predict(X)]
    linear_corr = pearsonr(linear_test_predict,Y.tolist())
    print (penguin_names[i],linear_corr[0])
    s.to_csv(penguin_names[i]+'_model_output.csv')
    
