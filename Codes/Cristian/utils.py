import pandas as pd
import numpy as np
from IPython.display import display, HTML

#https://amitness.com/2019/07/identify-text-language-python/#:~:text=Fasttext%20is%20an%20open-source,subword%20information%20and%20model%20compression.



def my_df_describe(df,name = 'dataset',show = True):
    print(20*'*'+name+20*'*')
    objects = []
    numerics = []
    for c in df:
        if (df[c].dtype == object):
            objects.append(c)
        else:
            numerics.append(c)
    
    df = df.replace(to_replace=['',' ','None','NaN'], value=np.nan)
    numeric_desc, categorical_desc = None,None

    if len(numerics)>0:
        numeric_desc = df[numerics].describe().transpose()
        counting = df.nunique().transpose()
        numeric_desc['unique'] = counting[numerics]
        numeric_desc['nulls'] = df[numerics].isna().sum().values
        numeric_desc['nulls_perc'] = df[numerics].isna().sum().values/df.shape[0]
        #print(df[numerics].isna().sum())
        #print(numeric_desc.shape)

    if len(objects)>0:
        categorical_desc = df[objects].describe().transpose()
        #print(df[objects].isna().sum())
        #print(categorical_desc.shape)
        categorical_desc['nulls'] = df[objects].isna().sum().values
        categorical_desc['nulls_perc'] = df[objects].isna().sum().values/df.shape[0]
    
    print('shape ',df.shape)
    if show:
        if len(numerics)>0:
            print(10*'*'+'numerics'+10*'*')
            display(numeric_desc)
        if len(objects)>0:
            print(10*'*'+'categorical'+10*'*')
            display(categorical_desc)

    return numeric_desc, categorical_desc

def clean_null(df_train,df_test,limit = 0):
    print('Train init shape ',df_train.shape)
    print('Test  init shape ',df_test.shape)
    df_train = df_train.replace(to_replace=['None','NaN'], value=np.nan)
    
    '''
    columns_ok = df_train.notna().any(axis=0)
    print('columns_ok ',columns_ok.sum())
    print('columns_no_ok ',len(df_train.columns)-columns_ok.sum())
    for name in df_train.columns:
        if not columns_ok[name]:
            print('has 0 elements -> deleting ..',name)
    df_train = df_train.loc[:,columns_ok]
    df_test = df_test.loc[:,columns_ok]
    '''
    columms_percent_nulls = df_train.isnull().mean()
    columns_ok = columms_percent_nulls < limit
    for name in df_train.columns:
        if not columns_ok[name]:
            print('has ',100*columms_percent_nulls[name] ,"%"+" null elements -> deleting ..",name)

    df_train = df_train.loc[:,columns_ok]
    df_test = df_test.loc[:,columns_ok]

    print('Train final shape',df_train.shape)
    print('Test  final shape',df_test.shape)
    
    return df_train,df_test

def process_countries(x):
    if type(x)!= str:
        return x
    if 'en:' in x:
        for element in x.split(' '):
            if not 'en:' in element:                
                return element.strip(',')
        for element in x.split(','):
            if not 'en:' in element:
                return element.strip(',')
        
    return x

def process_aditives(x):
    if type(x)!= str:
        return x
    texto = ''
    for additive in x.split(']  ['):
        for element in additive.split('->'):
            if not 'en:' in element:                
                for r in [' ',']','[']:
                    element = element.strip(r)                
                texto+=element+','
                break
    return texto
    
import fasttext
from deep_translator import GoogleTranslator as deepGoogleTranslator
from deep_translator import (
                             MicrosoftTranslator,
                             PonsTranslator,
                             LingueeTranslator,
                             MyMemoryTranslator,
                             YandexTranslator,
                             DeepL,
                             QCRI,
                             single_detection,
                             batch_detection)
from translate import Translator

class bnp_process_text:
    def __init__(self,path,target='en'):
        self.target = target
        PRETRAINED_MODEL_PATH = path
        self.model = fasttext.load_model(PRETRAINED_MODEL_PATH)
        self.to_language = None
    def get_language_fasttext(self,x):
        if x in ['None',float('nan'),np.nan]:
            return x
        elif type(x)!=str:
            return x
        else:
            sentences = [x]
            predictions = self.model.predict(sentences)

            return predictions[0][0][0].split('__')[-1]
    def set_model(self,model_name,x):
        self.to_language = x
        if model_name=='deepGoogleTranslator':
            self.translator  = deepGoogleTranslator(source=x, target=self.target)
        elif model_name=='LingueeTranslator':
            self.translator  = LingueeTranslator(source=x, target=self.target)
        elif model_name=='PonsTranslator':
            self.translator = PonsTranslator(source=x, target=self.target)
        elif model_name =='MyMemoryTranslator':
            self.translator  = MyMemoryTranslator(x, self.target)
        elif model_name == 'TextBlob':
            from textblob import TextBlob 
        elif model_name == 'translate':
            self.translator= Translator(from_lang=x,to_lang=self.target)
        elif model_name == 'googletrans':
            self.translator = Translator() 

    def get_language_deeptranslator(self,x):        

        if x in ['None',float('nan'),np.nan]:
            return x
        elif type(x)!=str:
            return x
        else:
            try:
                if model_name == 'TextBlob':
                    return blob.translate(to="en",)
                elif model_name == 'googletrans':
                    return translator.translate(x,src=self.to_language,dest=self.target).text
                return self.translator.translate(x)
            except:
                return x


