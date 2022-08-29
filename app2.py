# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 06:16:30 2022

@author: hp
"""



#path = r'C:\Users\hp\.spyder-py3\SMS_data.csv'
import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import re
import nltk
import spacy
import string
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter
from matplotlib import pyplot as plt

nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger') 
nltk.download('stopwords')  



            
            
            
            
           
            
    

     #df["Message_body"] = df["Message_body"].astype(str)


            
def main ():
          path = r'C:/Users/hp/.spyder-py3/SMS_data.csv'
     #df=pd.read_csv(path)
          full_df = pd.read_csv(path, encoding='unicode_escape')
     #full_df = pd.read_csv(path, encoding='unicode_escape')
          df = full_df[["Message_body"]]
          full_df.head( )     
          
          
          def remove_urls(text):
                          url_pattern = re.compile(r'[0-9]+')    
                          return url_pattern.sub('', text)
          def remove_numbers(text):
              url_pattern = re.compile(r'http\S+')
              return url_pattern.sub('', text)
           
          #full_df ["Message_body_updated"] = full_df["Message_body"].apply(lambda text: remove_numbers(text))
          lemmatizer = WordNetLemmatizer()
          wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV} 
          def lemmatize_words(text):
                        pos_tagged_text = nltk.pos_tag(text.split())
                        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
          
          PUNCT_TO_REMOVE = string.punctuation            
          def remove_punctuation(text):
     #"""custom function to remove the punctuation"""
             return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
          
          STOPWORDS = set(stopwords.words('english'))
          def remove_stopwords(text):
                 #"""custom function to remove the stopwords"""
                  return " ".join([word for word in str(text).split() if word not in STOPWORDS]) 
          full_df ["Message_body_updated"] = full_df["Message_body"].apply(lambda text: remove_punctuation(text))
          
          
          full_df["Message_body_updated"] = (
    
          full_df["Message_body"].str.lower() #LOWER CASED EVERYTHING

          .apply(lambda text: remove_urls(text)) #URLS REMOVED

          .apply(lambda text: remove_numbers(text)) #NUMBERS REMOVED

          .apply(lambda text: remove_punctuation(text)) #REMOVE PUNCTUATION

          .apply(lambda text: remove_stopwords(text)) #REMOVED STOP WORDS

          .apply(lambda text: lemmatize_words(text)) #LEMMATIZED TO ROOT WORD

          )
          
          def Word_Count(data_frame_col,counter_obj):
            for text in data_frame_col.values:
                 for word in text.split():
                    counter_obj[word] += 1
            return counter_obj
      
          cnt_most_common_Spam=Word_Count(full_df.query("Label == 'Spam'")['Message_body_updated'],Counter())
          cnt_most_common_Non_Spam=Word_Count(full_df.query("Label == 'Non-Spam'")['Message_body_updated'],Counter())
          
          
          print(cnt_most_common_Spam)
          print(cnt_most_common_Non_Spam)
          
          #
          df=pd.DataFrame(cnt_most_common_Spam, index=[0])
          
          md=pd.DataFrame(cnt_most_common_Non_Spam, index=[0])
          st.write('# SMS Classifier')
            
          st.header ("Created by : Raza Ali")
          
          
          
          full_df['Date_Received']=pd.to_datetime(full_df['Date_Received']) 
          full_df['Month']=full_df['Date_Received'].apply(lambda x: x.month_name())
          
          m=full_df['Month'].value_counts()
          dd= (
          pd.DataFrame(m,columns=['Month','Count']))
          
          st.subheader('Messages over Months')
          st.line_chart(data=m,)
         
          Label = st.sidebar.selectbox('Select label', full_df['Label'].unique())
          button = st.sidebar.button('show results')
          if Label =="Spam":
             df= (
             pd.DataFrame(cnt_most_common_Spam.most_common(15),columns=['Word','Count'])
            .sort_values('Count', ascending=1))
             st.sidebar.subheader('Common Words found in Spam')
             st.sidebar.bar_chart(data=df,x='Word',y='Count')           
                          
              
          else:
              md= (
              pd.DataFrame(cnt_most_common_Non_Spam.most_common(15),columns=['Word','Count'])
             .sort_values('Count', ascending=1))
              st.sidebar.subheader('Common Words found in Non-Spam')
              st.sidebar.bar_chart(data=md,x='Word',y='Count')
             
             
              
              
              
              
              
              
              
              
          
          
          
          
          
          
                      
      



             
          
          
          
          
          
          
          
          
          
          
          
        




     
if __name__=='__main__':
    main()
    
    
    
    
    