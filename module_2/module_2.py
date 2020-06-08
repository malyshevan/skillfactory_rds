#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind


# In[2]:


student_massiv = pd.read_csv('stud_math.xls')
student_massiv.head()
student_massiv.columns = ['school','sex','age','address','famsize','pstatus','medu','fedu','mjob','fjob',
'reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery',
'studytime_granular','higher','internet','romantic','famrel','freetime','goout','health','absences','score']
student_massiv.info()


# In[3]:


'''меняем в столбцах формата object NaN на нормальное значение'''
st_columns_object = {}
st_columns_not_object = []

st_columns = student_massiv.columns.tolist()


def object_col(x):
    if student_massiv.loc[:, x].dtype == 'object':
        st_columns_object.update({x:'No_information'})
    if student_massiv.loc[:, x].dtype != 'object':
        st_columns_not_object.append(x)

for i in st_columns:
    object_col(i)
    

student_massiv_1 = student_massiv.fillna(value=st_columns_object)
student_massiv_1.head()


# In[5]:


'''уберем NaN из score'''
student_massiv_1 = student_massiv_1.dropna(subset = ['score'])
print("Всего строк, без NaN:", student_massiv_1.score.value_counts().sum())
print("Уникальных значений:", student_massiv_1.score.nunique())


# In[ ]:


# student_massiv_2 = student_massiv_1.groupby(st_columns).count()
student_massiv_1.sort_values('score')


# In[ ]:


'''посмотрим статистические данные по полю absences, проверим на наличие выбросов'''
student_massiv_1.absences.hist()
student_massiv_1.absences.describe()


# In[ ]:


median = student_massiv_1.absences.median()
IQR = student_massiv_1.absences.quantile(0.75) - student_massiv_1.absences.quantile(0.25)
perc25 = student_massiv_1.absences.quantile(0.25)
perc75 = student_massiv_1.absences.quantile(0.75)
print('25-й перцентиль: {},'.format(perc25), '75-й перцентиль: {},'.format(perc75)
      , "IQR: {}, ".format(IQR),"Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))
student_massiv_1.absences.loc[student_massiv_1.absences.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].hist(bins = 5, range = (0, 100), 
                                                                                             label = 'IQR')
student_massiv_1.absences.loc[student_massiv_1.absences <= 100].hist(alpha = 0.5, bins = 5, range = (0, 100),
                                                        label = 'Здравый смысл')
plt.legend();


# In[ ]:


student_massiv_1 = student_massiv_1.loc[student_massiv_1.absences.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)]


# In[ ]:


'''проверяем столбец fedu'''
student_massiv_1.fedu.hist()
student_massiv_1.fedu.describe()


# In[ ]:


'''поле famrel'''
student_massiv_1.famrel.hist()
student_massiv_1.famrel.describe()


# In[ ]:


student_massiv_2 = student_massiv_1.replace({'fedu': {40.0:4.0},'famrel':{-1.0:1.0}})


# In[ ]:


'''проведем корреляционный анализ'''
student_massiv_3 = student_massiv_2[['age','absences','score']]
sns.pairplot(student_massiv_3, kind = 'reg')
# student_massiv_3


# In[ ]:


student_massiv_3.corr()


# In[ ]:


def get_boxplot(column):
    fig, ax = plt.subplots(figsize = (14, 4))
    sns.boxplot(x=column, y='score', 
                data=student_massiv_2.loc[student_massiv_1.loc[:, column].isin(student_massiv_2.loc[:, column].value_counts().index[:10])],
               ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + column)
    plt.show()


# In[ ]:


'''Анализ номинативных переменных'''
for col in st_columns_object.keys():
    get_boxplot(col)
for col in ['medu','fedu','traveltime','studytime','failures','studytime_granular','famrel',
'freetime','goout','health']:
    get_boxplot(col)


# In[ ]:


def get_stat_dif(column):
    cols = student_massiv_2.loc[:, column].value_counts().index[:30]    
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(student_massiv_2.loc[student_massiv_2.loc[:, column] == comb[0], 'score'], 
                        student_massiv_2.loc[student_massiv_2.loc[:, column] == comb[1], 'score']).pvalue \
            <= 0.05/len(combinations_all):
            print('Найдены статистически значимые различия для колонки', column)
            break


# In[ ]:


for col in st_columns_object.keys():
    get_stat_dif(col)
for col in ['medu','fedu','traveltime','studytime','failures','studytime_granular','famrel',
'freetime','goout','health']:
    get_stat_dif(col)


# In[ ]:


student_massiv_for_model = student_massiv_2.loc[:, ['age', 'absences', 'score', 'sex', 'address','higher','romantic','medu','failures']]
student_massiv_for_model.head()


# In[ ]:




