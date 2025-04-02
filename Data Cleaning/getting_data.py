#%% 
import pandas as pd
import numpy as np 
import sqlite3
# %%
df = pd.read_csv(r"ENTRADASEBZMORUMBI2025 - Entradas.csv")
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
        
    def to_date(self, date_column):
        self[date_column] = pd.to_datetime(self[date_column], format="%d/%m/%Y")
        return self
        
    def normalize_dollar(self, dollar_column):
        if dollar_column in self.columns:
            self[dollar_column] = self[dollar_column].apply(lambda x: x.replace("R$ ","").replace(",",".").replace(" ","").replace("0.0.0","0") if isinstance(x,str) else x)
            self[dollar_column] = self[dollar_column].astype(float)
        return self
    
    def to_boolean(self, dollar_column):
        if dollar_column in self.columns:
            self[dollar_column] = self[dollar_column].map({"NÃO": False, "SIM": True}).astype(bool)
        return self   
    
    def to_null(self, dollar_column):
        if dollar_column in self.columns:
            self[dollar_column] = self[dollar_column].apply(lambda x: np.nan if x == "VAZIO" else x)
        return self
    
    def rename_second_column(self):
        if len(self.columns) > 1:
            self.rename(columns={self.columns[0]: "produto"}, inplace=True)
            self.rename(columns={self.columns[1]: "vl_venda"}, inplace=True)
            self.rename(columns={self.columns[2]: "compra"}, inplace=True)
            self.rename(columns={self.columns[3]: "lucro"}, inplace=True)
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
print(new_df.shape)
new_df = new_df.assign(Bebidas=new_df["Bebidas"].str.split(",")).explode("Bebidas")
print(new_df.shape)
# %%
new_df["ID"].nunique()/new_df["Cliente"].nunique()
# %%
produtos = pd.read_csv(r"c:\Users\luiz\Documents\GitHub\EZmorumbi\Data Cleaning\ENTRADASEBZMORUMBI2025 - Produtos.csv")

# %%
dim_servico = produtos[["SERVIÇO","VALOR DO SERVIÇO"]]
dim_produto = produtos[["PRODUTO","VALOR DE VENDA","VALOR DE COMPRA","LUCRO"]]
dim_doce = produtos[["DOCE","VALOR DE VENDA.1","VALOR DE COMPRA.1","LUCRO.1"]]
dim_salgado = produtos[["SALGADO","VALOR DE VENDA.2","VALOR DE COMPRA.2","LUCRO.2"]]
dim_bebida = produtos[["BEBIDA","VALOR DE VENDA.3","VALOR DE COMPRA.3","LUCRO.3"]]
dim_funcionario = produtos[["FUNCIONARIO","PORCENTAGEM"]]
dim_taxa = produtos[["FUNCAO","TAXA"]]

#%%
dim_servico["tipo"] = "servico"
dim_doce["tipo"] = "doce"
dim_salgado["tipo"] = "salgado"
dim_bebida["tipo"] = "bebida"
dim_funcionario["tipo"] = "funcionario"
dim_taxa["tipo"] = "taxa"
# %%
dataframes = [dim_servico,dim_produto,dim_doce,dim_salgado,dim_bebida,dim_funcionario,dim_taxa]

dim_servico = data_cleaning(dim_servico)
dim_produto = data_cleaning(dim_produto)
dim_doce = data_cleaning(dim_doce)
dim_salgado = data_cleaning(dim_salgado)
dim_bebida = data_cleaning(dim_bebida)
#%%
dim_produto.head()
# %%
dim_produto.rename_second_column()
dim_doce.rename_second_column()
dim_salgado.rename_second_column()
dim_bebida.rename_second_column()
# %%
for data in dataframes:
    data.dropna(inplace=True)
#%%
dim_servico.rename(columns={
    "SERVIÇO" : "servico",
    "VALOR DO SERVIÇO" : "vl_venda"
})
# %%
dim_produto = dim_produto.normalize_dollar("lucro")
dim_doce = dim_doce.normalize_dollar("lucro")
dim_salgado = dim_salgado.normalize_dollar("lucro")
dim_bebida = dim_bebida.normalize_dollar("lucro")
dim_produto = dim_produto.normalize_dollar("vl_venda")
dim_doce = dim_doce.normalize_dollar("vl_venda")
dim_salgado = dim_salgado.normalize_dollar("vl_venda")
dim_bebida = dim_bebida.normalize_dollar("vl_venda")
dim_produto = dim_produto.normalize_dollar("compra")
dim_doce = dim_doce.normalize_dollar("compra")
dim_salgado = dim_salgado.normalize_dollar("compra")
dim_bebida = dim_bebida.normalize_dollar("compra")
# %%
tabela_fato = new_df[['data', 'ID', 'Colaborador', 'Cliente', 'Servico', 'ValordoServico',
       'ClienteNovo', 'Local', 'Produto', 'QuantidadeProduto',
       'ValorTotaldosProduto', 'Doces',
       'QuantidadeDoces', 'ValorTotaldosDoces', 'Salgados', 'QuantidadeSalgados',
       'Bebidas', 'QuantidadeBebidas',
       'ValorTotaldasBebidas', 'FormadePagamento', 'TaxaMaquina', 'Total(S+P)',
       'Total(S+P)*T', 'Total[(S+P)-LP]*T', 'TotalS+LP-Col',
       'TotalColaborador', 'Colaborador50%']]
#%%
tabela_fato = tabela_fato.rename(columns={
    'data': 'data',
    'ID': 'id',
    'Colaborador': 'colaborador',
    'Cliente': 'cliente',
    'Servico': 'servico',
    'ValordoServico': 'vl_servico',
    'ClienteNovo': 'cliente_novo',
    'Local': 'local',  
    'Produto': 'produto',
    'QuantidadeProduto': 'qtd_produto',
    'ValorTotaldosProduto': 'vl_total_produto',
    'Doces': 'doce',
    'QuantidadeDoces': 'qtd_doce',
    'ValorTotaldosDoces': 'vl_total_doces',
    'Salgados': 'salgado',  
    'QuantidadeSalgados': 'qtd_salgado',
    'ValorTotaldosSalgados': 'vl_total_salgado',
    'Bebidas': 'bebida',  
    'QuantidadeBebidas': 'qtd_bebida',
    'ValorTotaldasBebidas': 'vl_total_bebida',
    'FormadePagamento': 'forma_de_pagamento',
    'TaxaMaquina': 'tx_maquina',
    'Total(S+P)': 'total_sp',
    'Total(S+P)*T': 'total_spt',
    'Total[(S+P)-LP]*T': 'total_splt',
    'TotalS+LP-Col': 'total_slp_col',
    'TotalColaborador': 'total_colaborador',
    'Colaborador50%': 'colaborador_50'
})
#%%
tabela_fato = tabela_fato.rename(columns={
    'local': 'fidelizado'
})
#%%
dim_produto.head()
#%%
def to_string(dataframe,coluna):
    dataframe[coluna] = dataframe[coluna].astype(str)
    return dataframe[coluna].apply(lambda x: type(x)).value_counts()
#%%
to_string(dim_bebida,"produto")
#%%
to_string(dim_produto,"produto")
#%%
dim_produto = dim_produto[dim_produto["produto"] != "VAZIO"]
#%%
dataframes = [dim_bebida, dim_doce, dim_salgado]
for df in dataframes:
    df.dropna(subset=["vl_venda"],inplace=True)
#%%
dim_produto_2 = pd.concat([dim_bebida, dim_doce, dim_salgado])
#%%
dim_produto_2.head(100)
#%%
dim_p_b = dim_produto.merge(dim_produto_2, on="produto", how="left", suffixes=("", "n_"))
dim_p_b.drop(columns=["vl_vendan_","compran_","lucron_"],inplace=True)
#%%
dim_p_b.dropna(subset="vl_venda",inplace=True)
#%%
dim_p_b["tipo"] = dim_p_b["tipo"].apply(lambda x: "barbearia" if pd.isna(x) else x)
dim_p_b.head()
#%%
dim_p_b.drop_duplicates(subset=["produto"],inplace=True)
dim_p_b = feature_engineering(dim_p_b)
dim_p_b = dim_p_b.index_values("produto")
dim_p_b.head()
#%%
produto_map = dim_p_b[["produto","id_produto"]].to_dict()
#%%
tabela_fato["produto"] = tabela_fato["produto"].map(produto_map)
tabela_fato["doce"] = tabela_fato["doce"].map(produto_map)
tabela_fato["salgado"] = tabela_fato["salgado"].map(produto_map)
tabela_fato["bebida"] = tabela_fato["bebida"].map(produto_map)
# %%
tabela_fato.head(10)
# %%
df.info()
#%%
tabela_fato.info()
# %%
import mysql.connector
import pandas as pd

def create_table(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tabela_fato (
            data DATETIME,
            id DOUBLE,
            colaborador TEXT,
            cliente TEXT,
            servico TEXT,
            vl_servico DOUBLE,
            cliente_novo BOOLEAN,
            fidelizado TEXT,
            produto TEXT,
            qtd_produto TEXT,
            vl_total_produto DOUBLE,
            doce TEXT,
            qtd_doce DOUBLE,
            vl_total_doces DOUBLE,
            salgado TEXT,
            qtd_salgado DOUBLE,
            bebida TEXT,
            qtd_bebida TEXT,
            vl_total_bebida DOUBLE,
            forma_de_pagamento TEXT,
            tx_maquina DOUBLE,
            total_sp DOUBLE,
            total_spt DOUBLE,
            total_splt DOUBLE,
            total_slp_col DOUBLE,
            total_colaborador DOUBLE,
            colaborador_50 DOUBLE
        )
    ''')

def insert_data(cursor, df):
    for _, row in df.iterrows():
        cursor.execute('''
            INSERT INTO tabela_fato VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', tuple(row))

def main(df):
    conn = mysql.connector.connect(
        host='database-atinova.ct6oomqu6y49.sa-east-1.rds.amazonaws.com',
        user='integrantes',
        password='grupoPI2025',
        database='barbearia',  # Substituir pelo nome correto do banco de dados
        port=3306
    )
    cursor = conn.cursor()
    
    create_table(cursor)
    
    df.fillna('', inplace=True)  # Substituir NaN por strings vazias
    insert_data(cursor, df)
    
    conn.commit()
    cursor.close()
    conn.close()

# Chame a função passando seu DataFrame diretamente
main(tabela_fato)

#%%
