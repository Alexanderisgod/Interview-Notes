## Kaggle Tabular Playground Series 

[https://www.kaggle.com/code/alexandergod/tps-apr-lstms-attention/edit](https://www.kaggle.com/code/alexandergod/tps-apr-lstms-attention/edit)

### Step1——import Libraries

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


from sklearn.preprocessing import StandardScaler
from sklearn.metrics  import roc_auc_score
from sklearn.model_selection import GroupKFold

import os
# filter lowe level: like warnings sth.
tf.get_logger.setLevel("Error")

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callback import EarlyStoping, ModelCheckpoing, ReduceLROnPlateau
from tensorflow.keras.layers import *

np.random.seed(42)
tf.random.set_seed(42)

train=pd.read_csv("train_path")
train_label=pd.read_csv("train_label")
test=pd.read_csv("test_path")
test_label=pd.read_csv("test_label")
```

```python
# get the lag data
def preprocessiong(df):
    for feature in features:
        df[feature+'_lag1']=df.groupby("sequence")[feature].shift(1)
        df.fillna(0, inplace=True)
        df.[feature+'_diff1']=df[feature]-df[feature+'_lag1']
```

```python
# get the model

from keras_self_attention import SeqSelfAttention

def lstm_att_model():
    x_input=Input(shape=(train.shape[-2:]))
    
    x=Bidirectional(LSTM(512, return_sequence=True))(x_input)
    x=Bidirectional(LSTM(384, return_sequence=True))(x)
    x=SeqSelfAttention(attention_activation="sigmoid", name='attention_weight')(x)
    x=GlobalAveragePooling1D(x)
    
    x_output=Dense(units=1, activation='sigmoid')(x)
    
    model=Model(inputs=x_input, outputs=x_output, name='lstm_model')
    
    return model

model= lstm_att_model()
```



==How to train a model from scratch??==

```python
BATCH_SIZE=64
VERBOSE=False

predictions, scores=[], []

k=GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(k.split(train,labels, groups.unique())):
    print('-'*15, '>', f'Fold:{fold+1}', '<', '-'*15)
    x_train, x_val = train[train_idx], train[val_idx]
    y_train, y_val = labels.iloc[train_idx].values, labels.iloc[val_idx].values
    
    
    mdoel=lstm_attn_model()
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics='AUC')
    
    lr=ReduceLROnPlateau(monitor='val_auc', factor=0.5, 
                        patience=2, verbose=VERBOSE, mode='max')\
    es=EarlyStopping(monitor='val_auc', patience=7, verbose=VERBOSE,
                     mode='max', restore_best=Ture)
    chk_point = ModelCheckpoint(f'./TPS_model_2022_{fold+1}C.h5', 
                                monitor='val_auc', verbose=VERBOSE, 
                                save_best_only=True, mode='max')
    model.fit(x_train, y_train,
             validation_data=(x_val, y_val),
             epochs=20,
             verbose=VERBOSE,
             batch_size=BATCH_SIZE,
             callback=[lr, chk_point, es])
    model = load_model(f'./TPS_model_2022_{fold+1}C.h5', 
                        custom_objects=SeqSelfAttention.get_custom_objects())
    y_pred=model.predict(x_val, batch_size=BATCH_SIZE).squeeze()
    score=roc_auc_score(y_val, y_pred)
    scores.append(score)
    predictions.append(model.predict(test, batch_size=BATCH_SIZE).squeeze())
    

```

