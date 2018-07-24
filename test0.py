import pickle
import pandas as pd



df = pd.read_pickle('countfeature.pickle',compression='gzip')


print(df.shape)
df.info()


