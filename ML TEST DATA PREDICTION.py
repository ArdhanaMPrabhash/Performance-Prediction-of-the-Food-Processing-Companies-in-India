#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#prediction using median....Random forest regressor


# In[84]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[87]:


df_test=pd.read_csv('C:\\Users\\Ardhana\\Downloads\\test_data.csv')


# In[88]:


df_test


# In[89]:


df_test.info()


# In[90]:


missing_value_dftest= df_test.isnull()
print(missing_value_dftest)


# In[91]:


missing_counttest =missing_value_dftest.sum()
print(missing_counttest)


# In[92]:


df_test.describe()


# In[93]:


df1 = df_test.drop(['Company Name', 'Year'], axis =1)
df1


# In[94]:


df1.mean()


# In[95]:


df1.median()


# In[96]:


dfmed =df1.fillna(df1.median())
dfmed


# In[97]:


#USING TRAINED MODEL PREDICTING THE TEST LABELS


# # USING median imputation....PREDICTIONS...

# In[98]:


from joblib import load

# Load the trained model
model = load('C:\\Users\\Ardhana\\Downloads\\DEVMED.csvtraining_modelmed.joblib')
model


# In[99]:


dfmed


# In[100]:


# Make predictions on the test data
predictions2 = model.predict(dfmed)
predictions2


# In[103]:


import numpy as np 
import pandas as pd 
   
  
#convert numpy array to dataframe  

predicted_y = pd.DataFrame(predictions2) 
print("\nPandas DataFrame: ") 
predicted_y


# In[104]:


import pandas as pd

predicted_y.to_csv("Predicted labels.csv", index=False)
from IPython.display import FileLink
FileLink("Predicted labels.csv")

