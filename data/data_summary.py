import pandas as pd
import os

path = 'label_instruction/label_instruction.csv'

df = pd.read_csv(path, dtype={'curve': object})

def counter(df,type):
    df = df[df['type'] == type]
    curve = df['curve']
    curve = curve.values
    ok = 0
    cu = 1
    for value in curve:
        ok += value.count('0')
        cu += value.count('1')
    return ok, cu

types = ['U100','U150','A30']
for typ in types:
    print(typ)
    ok, cu = counter(df, typ)
    print(f'ok: {ok}')
    print(f'cu: {cu}')
    print(ok+cu)
