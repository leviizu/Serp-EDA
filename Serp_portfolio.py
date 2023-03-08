#!/usr/bin/env python
# coding: utf-8

# # Serp Flight Data Analysis

# ## Project Purpose:
# 

# To demonstrate subject mattaer expertise of serp and seo analysis through this analytics project

# ## Business Case:

# - To determine Which are the top searched domains globally and in the US and UK
# - To determine what factors could increase or decrease keyword rankings 
# - To understand the keyword ranks per percentage for Top domains in the different markets
# - To ascertain if lenght of keywords, snippet or title affects flight search ranking 
# - to ascertai if title structure affect ranking

# In[70]:


# importing libraries
import pandas as pd
import glob
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
import plotly


# we have serp results from 24 periods 
# merging the files
joined_files = os.path.join("/Users/levy/Documents/serp", "flights_tickets_serp*.csv")
  
# A list of all joined files is returned
joined_list = glob.glob(joined_files)

#making all columns visible
pd.set_option('display.max_columns', 26)

import warnings
warnings.filterwarnings('ignore')
  
# Finally, the files are joined
serp = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
serp.head(5)


# In[71]:


serp.sort_values(by=['queryTime'], ascending=False)


# In[72]:


#checking for nulls and datatypes
serp.info()


# In[73]:


serp.describe(include=object)


# In[74]:


serp.describe()


# ### Data cleaning

# In[75]:


#checking/verifying  if there are  missing values 
serp.isnull().sum()


# In[76]:


#dropping columns that have null values and will be not needed for analysis
serp.dropna(axis=0, inplace=True)
serp.head()


# In[77]:


#checking for duplicates
serp.duplicated().value_counts()


# In[78]:


serp.rename(columns={'gl':'Location'}, inplace=True)
serp.head()


# In[79]:


#setting the correct data types for query time
serp['queryTime']= pd.to_datetime(serp['queryTime'])
serp['queryTime'].dtype


# ### Feature Engineering

# In[80]:


#formatting data
#query_time contains date and time, we will be creating new field with only dates
#will rename 'gl' to 'location' for easy comprehension
#some titles contain prices, will extract and create new field for prices and convert them to a single currency
serp['QueryDate']=serp['queryTime'].dt.date
serp.head()


# In[81]:


#Extracting prices and currency from title
serp['price']=(serp['title'].str.extract('[$£](\d+,?\d+\.?\d+)')[0].str.replace(',', '').astype(float))
serp['currency'] = serp['title'].str.extract('([$£])')

serp.head(10)


# In[82]:


#grouping the dates into period of months
#creating and storing the information in a field
serp['period']=serp['queryTime'].dt.to_period('M').sort_values(ascending=True)


# In[224]:


#Since we have searches from two countries, we will be creating subsets for both countries
serp_us = serp[serp['Location'] == 'us'] # create a subset for US
serp_uk = serp[serp['Location'] == 'uk'] # create a subset for UK


# ## EDA

# In[113]:


#which domains have the most Global market shares so far?
#will be calculating weighted and non-weighted market share 
#Non weighted market share(number of searches/total number of searches)
#weighted market share(search volume/Total search volume)

#creating a subset for the analysis
serp_highest_share=serp[['displayLink','rank']]

#calculating %marketshare(% of number of searches per domain name/total number of searches)
Newcolumn=serp_highest_share['displayLink'].value_counts()/serp_highest_share['displayLink'].count()*100

#extracting top 10 domains with most searches
Newcolumn_second=serp_highest_share['displayLink'].value_counts().head(10)

#Aggregating
serp_highest_share_result=serp_highest_share.groupby(['displayLink']).mean().round(1).assign(MarketShare=Newcolumn.head(10),NumberOfSearches=Newcolumn_second)

#sorting and editing for readability
serp_highest_share_result=serp_highest_share_result.sort_values(by=['MarketShare'],ascending=False).head(10)
serp_highest_share_result['MarketShare']=serp_highest_share_result['MarketShare'].round(1).astype('str') + '%'
serp_highest_share_result


# In[227]:


#which domains have the most market shares so far in the US?
#creating a subset from the US subset
serp_highest_share=serp_us[['displayLink','rank']]

#calculating %marketshare(% of number of searches per domain name/total number of searches)
Newcolumn=serp_highest_share['displayLink'].value_counts()/serp_highest_share['displayLink'].count()*100

#extracting top 10 domains with most searches
Newcolumn_second=serp_highest_share['displayLink'].value_counts().head(10)

#Aggregating 
serp_highest_share_result_us=serp_highest_share.groupby(['displayLink']).mean().round(1).assign(MarketShare=Newcolumn.head(10),NumberOfSearches=Newcolumn_second).sort_values(by=['MarketShare'],ascending=False).head(10).reset_index()

#adding % for readability
serp_highest_share_result_us['MarketShare']=serp_highest_share_result_us['MarketShare'].round(1).astype('str') + '%'

serp_highest_share_result_us


# In[226]:


#Top domains in UK by marketshare
serp_highest_share_result_us['displayLink']=serp_highest_share_result_us['displayLink'].str.replace("www."," ")
plt.bar('displayLink','MarketShare', data=serp_highest_share_result_us,)
plt.xticks(rotation = 45)
plt.ylabel('Market share (%)')


# In[ ]:





# In[228]:


#which domains have the most market shares so far in the UK?
#creating a subset from the Uk subset
serp_highest_share=serp_uk[['displayLink','rank']]

#calculating %marketshare(% of number of searches per domain name/total number of searches)
Newcolumn=serp_highest_share['displayLink'].value_counts()/serp_highest_share['displayLink'].count()*100

#extracting top 10 domains with most searches
Newcolumn_second=serp_highest_share['displayLink'].value_counts().head(10)

#Aggregating 
serp_highest_share_result_uk=serp_highest_share.groupby(['displayLink']).mean().round(1).assign(MarketShare=Newcolumn.head(10),NumberOfSearches=Newcolumn_second).sort_values(by=['MarketShare'],ascending=False).head(10).reset_index()

#sorting and editing for readability
serp_highest_share_result_uk['MarketShare']=serp_highest_share_result_uk['MarketShare'].round(1).astype('str') + '%'
serp_highest_share_result_uk


# In[229]:


#plotting results on a bar chart
plt.rcParams['figure.figsize'] = (12,4)
serp_highest_share_result_uk['displayLink']=serp_highest_share_result_uk['displayLink'].str.replace("www."," ")
plt.bar('displayLink','MarketShare', data=serp_highest_share_result_uk)
plt.xticks(rotation = 45)
plt.ylabel('Market share (%)')


# In[ ]:





# In[120]:


#How well has each of  the Top 9 domains by market share ranked over time?
#Analysing the UK subset

#extracting the top 9 domains and adding them to a list
rank_result=serp_highest_share_result_uk.reset_index().head(9)
top_nine_domains_uk=[]
for domain in rank_result['displayLink']:
    top_nine_domains_uk.append(domain)
    
#extracting data for the top 9 domains in the Uk from the serp dataset
highest_rank_result_uk_=serp.query("displayLink in @top_nine_domains_uk and Location =='uk'")
highest_rank_result_uk_

highest_rank_result_uk=highest_rank_result_uk_[['displayLink','rank','QueryDate']]

#finding the average rank of these domains over time
rank_result_uk=highest_rank_result_uk.groupby(['displayLink','QueryDate']).mean().reset_index()


#plotting results
g = sns.FacetGrid(rank_result_uk, col="displayLink",col_wrap=3,height=3.2,aspect=1.9,margin_titles=True,despine=False)
g.map_dataframe(sns.lineplot, x="QueryDate",y="rank", marker='o',color='b')
plt.title('Company')

#reversing the y-axis since 1 is meant to be the peak of the ranking
plt.gca().invert_yaxis()

#adding a line showing median rank across all domains
g.refline(y=rank_result_uk["rank"].median())

g.set_axis_labels("Date", "Rank", color='b')
g.set_titles(col_template="{col_name}", row_template="{row_name}",size=13, color='black')
g.set(ylim=(10, 1), yticks=[1, 3, 5, 8, 10])

plt.show()


# In[118]:


#Analysing the US subset
#extracting the top 9 domains and adding them to a list
rank_result_us=serp_highest_share_result_us.reset_index().head(9)
top_nine_domains_us=[]
for domain in rank_result_us['displayLink']:
    top_nine_domains_us.append(domain)
    
#extracting data for the top 9 domains in the Uk from the serp dataset
highest_rank_result_us_=serp.query("displayLink in @top_nine_domains_us and Location =='us'")
highest_rank_result_us=highest_rank_result_us_[['displayLink','rank','QueryDate']]

#finding the average rank of these domains over time
rank_result_us=highest_rank_result_us.groupby(['displayLink','QueryDate']).mean().reset_index()


#plotting results
g = sns.FacetGrid(rank_result_us, col="displayLink",col_wrap=3,height=3.2,aspect=1.9,margin_titles=True,despine=False)
g.map_dataframe(sns.lineplot, x="QueryDate",y="rank", marker='o',color='b')
plt.title('Company')

#reversing the y-axis since 1 is meant to be the peak of the ranking
plt.gca().invert_yaxis()

#adding a line showing median rank across all domains
g.refline(y=rank_result_uk["rank"].median())

g.set_axis_labels("Date", "Rank", color='b')
g.set_titles(col_template="{col_name}", row_template="{row_name}",size=13, color='black')
g.set(ylim=(10, 1), yticks=[1, 3, 5, 8, 10])

plt.show()


# In[121]:


#Understanding and categorizing the keyword ranks of top domains in the UK
#how do keywords rank? How many percent of Keyword searches rank in the first three or first ten

#defining a function that will categorise these keyword ranks
def Categorize_rank(x):
    if x<=3:
        return 'Hyper Traffic(1-3)'
    else:
        return 'Traffic(4-10)'
    
#applying the categorize_rank function    
highest_rank_result_uk_['rank_category']=highest_rank_result_uk_['rank'].apply(Categorize_rank)

cat_result=highest_rank_result_uk_.groupby(['displayLink','rank_category',]).count().reset_index()

cat_result=cat_result.pivot(index='displayLink',columns='rank_category',values='rank').reset_index()

#handling nan values
cat_result['Hyper Traffic(1-3)']=cat_result['Hyper Traffic(1-3)'].replace(np.nan, 0)

#calculating percentages of rank categories
cat_result['Hyper Traffic(1-3)%']=cat_result['Hyper Traffic(1-3)']/(cat_result['Traffic(4-10)']+cat_result['Hyper Traffic(1-3)'])*100

#converting to wholenumber 
cat_result['Hyper Traffic(1-3)%']=cat_result['Hyper Traffic(1-3)%'].round(0)

#calculating percentages of rank categories
cat_result['Traffic(4-10)%']=cat_result['Traffic(4-10)']/(cat_result['Hyper Traffic(1-3)']+cat_result['Traffic(4-10)'])*100

#converting to wholenumber 
cat_result['Traffic(4-10)%']=cat_result['Traffic(4-10)%'].round(0)

#sorting by Hyper Traffic(1-3)%
cat_result_uk=cat_result.sort_values(by=['Hyper Traffic(1-3)%'], ascending=False)
cat_result_uk


# In[122]:


#Understanding and categorizing the keyword ranks of top domains in the UK
#how do keywords rank? How many percent of Keyword searches rank in the first three or first ten

#defining a function that will categorise these keyword ranks
def Categorize_rank(x):
    if x<=3:
        return 'Hyper Traffic(1-3)'
    else:
        return 'Traffic(4-10)'
    
#applying the categorize_rank function    
highest_rank_result_us_['rank_category']=highest_rank_result_us_['rank'].apply(Categorize_rank)

cat_result=highest_rank_result_us_.groupby(['displayLink','rank_category',]).count().reset_index()

cat_result=cat_result.pivot(index='displayLink',columns='rank_category',values='rank').reset_index()

#handling nan values
cat_result['Hyper Traffic(1-3)']=cat_result['Hyper Traffic(1-3)'].replace(np.nan, 0)

#calculating percentages of rank categories
cat_result['Hyper Traffic(1-3)%']=cat_result['Hyper Traffic(1-3)']/(cat_result['Traffic(4-10)']+cat_result['Hyper Traffic(1-3)'])*100

#converting to wholenumber 
cat_result['Hyper Traffic(1-3)%']=cat_result['Hyper Traffic(1-3)%'].round(0)

#calculating percentages of rank categories
cat_result['Traffic(4-10)%']=cat_result['Traffic(4-10)']/(cat_result['Hyper Traffic(1-3)']+cat_result['Traffic(4-10)'])*100

#converting to wholenumber 
cat_result['Traffic(4-10)%']=cat_result['Traffic(4-10)%'].round(0)

#sorting by Hyper Traffic(1-3)%
cat_result_us=cat_result.sort_values(by=['Hyper Traffic(1-3)%'], ascending=False)
cat_result_us


# In[93]:


#How many searches do the top 10 most occuring domains have appearing in each rank in the US?

#filtering and extracting these Top 10 domains
top_domains_rank = serp[serp['Location']=='us']
top_domains_rank = top_domains_rank['displayLink'].value_counts().head(10).index.tolist()
top_serp_flights = serp[serp['displayLink'].isin(top_domains_rank)]

rank_counts_flights = top_serp_flights.groupby(['displayLink', 'rank']).agg({'rank': ['count']}).reset_index()
rank_counts_flights.columns = ['displayLink', 'rank', 'count']
rank_counts_flights.head()

#plotting results
fig = go.FigureWidget()

#labelling axis
fig.add_scatter(x=top_serp_flights['displayLink'].str.replace('www.', ''),
                y=top_serp_flights['rank'], mode='markers',
                marker={'size': 35, 'opacity': 0.035,})

#insering variables
fig.add_scatter(x=rank_counts_flights['displayLink'].str.replace('www.', ''),
                y=rank_counts_flights['rank'], mode='text', text=rank_counts_flights['count'])

fig.layout.hovermode = False
fig.layout.yaxis.autorange = 'reversed'
fig.layout.yaxis.zeroline = False
fig.layout.yaxis.tickvals = list(range(1, 11))
fig.layout.height = 600
fig.layout.title = 'Top Domains for Flights and Tickets Keywords - Google - USA'
fig.layout.yaxis.title = 'SERP Rank (number of appearances)'
fig.layout.showlegend = False
fig.layout.paper_bgcolor = '#eeeeee'
fig.layout.plot_bgcolor = '#eeeeee'
iplot(fig)
#This chart cannot be rendered on Github, it will be downloaded and uploaded


# In[94]:


#How many searches do the top 10 most occuring domains have appearing in each rank in the UK?

#filtering and extracting these Top 10 domains
top_domains_rank = serp[serp['Location']=='uk']
top_domains_rank = top_domains_rank['displayLink'].value_counts().head(10).index.tolist()
top_serp_flights = serp[serp['displayLink'].isin(top_domains_rank)]

rank_counts_flights = top_serp_flights.groupby(['displayLink', 'rank']).agg({'rank': ['count']}).reset_index()
rank_counts_flights.columns = ['displayLink', 'rank', 'count']
rank_counts_flights.head()

#plotting results
fig = go.FigureWidget()

#labelling axis
fig.add_scatter(x=top_serp_flights['displayLink'].str.replace('www.', ''),
                y=top_serp_flights['rank'], mode='markers',
                marker={'size': 35, 'opacity': 0.035,})

#insering variables
fig.add_scatter(x=rank_counts_flights['displayLink'].str.replace('www.', ''),
                y=rank_counts_flights['rank'], mode='text', text=rank_counts_flights['count'])

fig.layout.hovermode = False
fig.layout.yaxis.autorange = 'reversed'
fig.layout.yaxis.zeroline = False
fig.layout.yaxis.tickvals = list(range(1, 11))
fig.layout.height = 600
fig.layout.title = 'Top Domains for Flights and Tickets Keywords - Google - USA'
fig.layout.yaxis.title = 'SERP Rank (number of appearances)'
fig.layout.showlegend = False
fig.layout.paper_bgcolor = '#eeeeee'
fig.layout.plot_bgcolor = '#eeeeee'
iplot(fig)
#This chart cannot be rendered on Github, it will be downloaded and uploaded


# In[95]:


#Visualizing Keyword ranks as per percentage of total search for each domain in the UK Market

cat_result_plot=cat_result_uk[['displayLink','Hyper Traffic(1-3)%','Traffic(4-10)%']]
cat_result_plot.plot(kind='bar',x='displayLink',
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))

plt.legend(loc="upper left", ncol=2)
plt.xlabel("domain")
plt.ylabel("Proportion")
plt.show()


# In[96]:


#Visualizing Keyword ranks as per percentage of total search for each domain in the US Market

cat_result_plot=cat_result_us[['displayLink','Hyper Traffic(1-3)%','Traffic(4-10)%']]
cat_result_plot.plot(kind='bar',x='displayLink',
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))

plt.legend(loc="upper left", ncol=2)
plt.xlabel("domain")
plt.ylabel("Proportion")
plt.show()


# In[97]:


#most searched phrases in the US#
mostSearchedKeywords=serp.query("Location =='us'")
mostSearchedKeywords=mostSearchedKeywords['searchTerms'].value_counts().head(10)
mostSearchedKeywords


# In[98]:


#most expensive flight advert 
mostExpensiveFlights_us=serp.query("Location =='us'")
mostExpensiveFlights_us=mostExpensiveFlights_us.groupby(['searchTerms','displayLink','rank'])[['price']].max()
mostExpensiveFlights_us.sort_values(by='price', ascending=False).head(10)


# In[99]:


#most expensive  flight searches in uk
mostExpensiveFlights_uk=serp.query("Location =='uk'")
mostExpensiveFlights_uk=mostExpensiveFlights_uk.groupby(['searchTerms','displayLink','rank'])[['price']].max()
mostExpensiveFlights_uk.sort_values(by='price', ascending=False).head(10)


# In[100]:


#most expensive  flight searches in us
mostExpensiveFlights_us=serp.query("Location =='us'")
mostExpensiveFlights_us=mostExpensiveFlights_us.groupby(['searchTerms','displayLink','rank'])[['price']].max()
mostExpensiveFlights_us.sort_values(by='price', ascending=False).head(10)


# In[146]:


#Visualizing the prices distribution across US top 10 domains 
top_domains_rank = serp[serp['Location']=='us']
top_ten_us = top_domains_rank['displayLink'].value_counts().head(10).index.tolist()
top_ten_domains_us=[]
for domain in top_ten_us:
    top_ten_domains_us.append(domain)
serp_prices_uk=serp.query("displayLink in @top_ten_domains_us and Location =='us'")
sns.stripplot(data=serp_prices_uk, x="displayLink", y="price",
    jitter=False, s=20, marker="D", linewidth=1, alpha=.1)
plt.xticks(rotation = 45)


# In[147]:


#Visualizing the prices distribution across UK top 10 domains 
top_domains_rank = serp[serp['Location']=='uk']
top_ten_uk = top_domains_rank['displayLink'].value_counts().head(10).index.tolist()
top_ten_domains_uk=[]
for domain in top_ten_uk:
    top_ten_domains_uk.append(domain)
serp_prices_uk=serp.query("displayLink in @top_ten_domains_uk and Location =='uk'")
sns.stripplot(data=serp_prices_uk, x=["displayLink", y="price",
    jitter=False, s=20, marker="D", linewidth=1, alpha=.1)
plt.xticks(rotation = 45)


# In[103]:


#Checking the mean rank of Searches without price tags

serp_us=serp.query("Location=='us'")
no_prices=serp_us[serp_us['price'].isnull()]
no_prices['rank'].mean()


# In[104]:


#Checking the mean rank of Searches with price tags

serp_us=serp.query("Location=='us'")
no_prices=serp_us[serp_us['price'].notnull()]
no_prices['rank'].mean()


# In[105]:


#Checking the mean rank of Searches with price tags

serp_us=serp.query("Location=='uk'")
no_prices=serp_us[serp_us['price'].notnull()]
no_prices['rank'].mean()


# In[106]:


#Checking the mean rank of Searches without price tags
serp_us=serp.query("Location=='uk'")
no_prices=serp_us[serp_us['price'].isnull()]
no_prices['rank'].mean()


# #### Searches whose title have price tags have higher ranks in the US & UK

# # Content quantity analysis
# Since 'totalresults' shows how many pages each keyword has that are eligible to appear for a specific keyword.  
# Let's see what keywords  have the most pages
# Flight to HongKong has most pages 

# In[107]:


#Keyword with Most results
MostSearchedWord=serp[['totalResults','searchTerms' ]]
MostSearchedWord.groupby(['searchTerms']).sum().sort_values(by='totalResults', ascending=False).head(10)


# ##### Keyword concentration in Titles and snippets
# As we have seen before a slightly better ranking for a search term makes a huge difference in the traffic for the website. To jump between the second and the first rank doubles the traffic, and this can also double the profit. Therefore, we will have a look at the optimal quantity of keywords in the title and in the snippet to get a better ranking.
# 
# 

# In[108]:


#checking the titles
with pd.option_context('display.max_colwidth', 200):
  print(serp['title'])


# In[ ]:





# In[109]:


#does the number of words in the title or snippet affect the ranking?
serp['no_of_words_title']=serp['title'].replace(np.nan, 'nosnippet').apply(lambda x: len(x.split(" ")))
serp['no_of_words_snippet']=serp['snippet'].replace(np.nan, 'nosnippet').apply(lambda x: len(x.split(" ")))
serp['no_of_words_search_item']=serp['searchTerms'].replace(np.nan, 'nosnippet').apply(lambda x: len(x.split(" ")))

# Calculate the % of keywords/search terms in the titles
serp["%word_conc_in_title"] = serp['no_of_words_search_item']/serp['no_of_words_title']
serp["%word_conc_in_title"] = 100 * serp["%word_conc_in_title"] # Convert the result in %

# Calculate the % of keywords/search terms in the snippet
serp["%word_conc_in_snippet"] = serp['no_of_words_search_item']/serp['no_of_words_snippet']
serp["%word_conc_in_snippet"] = 100 * serp["%word_conc_in_snippet"] # Convert the result in %


# In[110]:


# Display the result
plot_snippet_rank = pd.pivot_table(serp, values = "%word_conc_in_snippet", index = "rank", aggfunc = "mean").sort_index(ascending = False)
plot_snippet_rank.plot.barh(figsize = (8,5), color = (0.32, 0.32, 0.5))
plt.legend("")
plt.xlabel("Exact keyword concentration in %", fontsize = 12)
plt.ylabel("# rank", fontsize = 14)
plt.title("Average keyword concentration in snippets\nby rank", fontsize = 16)
plt.show()


# In[111]:


serp.corr(method='spearman')


# In[112]:


serp.corr(method='spearman')['rank'].sort_values(ascending=False)


# - Percentage word concentration and number of word in snippet seems to be correlated to ranking
# - The Lower the number of words per key phrase the higher the ranking

# In[ ]:





# In[ ]:




