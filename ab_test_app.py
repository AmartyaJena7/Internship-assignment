#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
from Assignment_Solution import ab_test  

def main():
    st.title('A/B Test Hypothesis Testing App')
    
    st.sidebar.header('Input Parameters')
    
    control_visitors = st.sidebar.number_input('Control Group Visitors', min_value=0, value=0)
    control_conversions = st.sidebar.number_input('Control Group Conversions', min_value=0, value=0)
    treatment_visitors = st.sidebar.number_input('Treatment Group Visitors', min_value=0, value=0)
    treatment_conversions = st.sidebar.number_input('Treatment Group Conversions', min_value=0, value=0)
    
    confidence_level = st.sidebar.selectbox('Confidence Level', [90, 95, 99])
    
    if st.sidebar.button('Run A/B Test'):
        result = ab_test(control_visitors, control_conversions, treatment_visitors, treatment_conversions, confidence_level)
        st.write('Result of A/B Test:', result)

if __name__ == '__main__':
    main()


# In[ ]:




