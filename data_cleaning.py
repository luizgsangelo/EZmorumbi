#%% 
import pandas as pd
import numpy as np 
import sqlite3

# %%
df = pd.read_csv(r"ENTRADASEBZMORUMBI2025 - Entradas.csv")
df.head()
# %%
df.info()
# %%
df = df.dropna(subset=("Colaborador"))
df.shape
# %%
df.columns
# %%
df.isna().sum()
# %%
df.drop(columns=["Unnamed: 48","Unnamed: 49"],inplace=True)
#%% 
df.columns
# %%
df.rename(columns={' ': "data"}, inplace=True)
# %%
class data_cleaning(pd.DataFrame):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def to_date(self,date_column):
        self[date_column] = pd.to_datetime(self[date_column], format="%d/%m/%Y")
        return self
        
    def normalize_dollar(self,dollar_column):
        if dollar_column in self.columns:
            self[dollar_column] = self[dollar_column].apply(lambda x: x.replace("R$ ","").replace(",",".").replace(" ","").replace("0.0.0","0") if isinstance(x,str) else x)
            self[dollar_column] = self[dollar_column].astype(float)
        return self
    
    def to_boolean(self,dollar_column):
        if dollar_column in self.columns:
            self[dollar_column] = self[dollar_column].map({"NÃO": False, "SIM": True}).astype(bool)
        return self   
    
    def to_null(self,dollar_column):
        if dollar_column in self.columns:
            self[dollar_column] = self[dollar_column].apply(lambda x: np.nan if x == "VAZIO" else x)
        return self
    
class feature_engineering(data_cleaning):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
    def index_values(self,column):
        unique_values = self[column].unique()
        dicionario = {value: key for key, value in enumerate(unique_values)}
        self[f"id_{column}"] = self[column].map(dicionario)
        return self
    
# %%
new_df = data_cleaning(df)
new_df = new_df.to_date("data")
new_df.info()
# %%
colunas_valor = [col for col in df.columns if col.startswith("Valor")] + ["TaxaMaquina","Total(S+P)","Total(S+P)*T","Total[(S+P)-LP]*T","TotalS+LP-Col","TotalColaborador","Colaborador50%"] + [col for col in df.columns if col.startswith("AuxValor")] + [col for col in df.columns if col.startswith("AuxDesconto")]
for col in colunas_valor:
    new_df = new_df.normalize_dollar(col)
new_df.info()

# %%
colunas_valor = [col for col in df.columns if col.startswith("AuxValor")]
for col in colunas_valor:
    new_df = new_df.normalize_dollar(col)
new_df.info()
# %%
colunas_valor = [col for col in df.columns if col.startswith("AuxDesconto")]
for col in colunas_valor:
    new_df = new_df.normalize_dollar(col)
new_df.info()
# %%
new_df.head()
# %%
new_df = new_df.to_boolean("ClienteNovo")
new_df.info()
#%% 
new_df.head()
# %%
colunas = ["Produto","Doces","Salgados","Bebidas"]
for col in colunas:
    new_df = new_df.to_null(col)
new_df.head()
# %%
new_df.isnull().sum()
# %%
new_df = feature_engineering(new_df)
new_df = new_df.index_values("Servico")
new_df.info()
# %%
print(new_df.count())
new_df = new_df.assign(Bebidas=new_df["Bebidas"].str.split(",")).explode("Bebidas")
print(new_df.count())
#%% 
new_df['month'] = new_df['data'].dt.to_period('M')
grouped_df  = new_df.groupby("data").agg(Pedidos = ("ID","nunique"),
                            Valor_total = ("ValordoServico","sum"),
                            clientes = ("Cliente","nunique")).reset_index()
import matplotlib.pyplot as plt 
plt.figure(figsize=(12,10))
plt.plot(grouped_df["data"].astype(str), grouped_df["clientes"], label = "clientes")
plt.plot(grouped_df["data"].astype(str), grouped_df["Pedidos"], label = "pedidos")
plt.xticks(rotation=90)
plt.legend()
# %%
new_df["ID"].nunique()/new_df["Cliente"].nunique()
# %%
