import pandas as pd
import numpy as np
import json
import umap
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import os

class AnalysisData:
    def __init__(self, file_json, file_csv):
        self.file_json = file_json
        self.file_csv = file_csv

# Методы

    def r_json(self):
        try:
            with open(self.file_json, "r", encoding="utf-8") as f:  
                data = json.load(f)  
                df_json=pd.DataFrame(data["hits"]["hits"])
                df_json["vector"] = df_json["vector"].apply(lambda x: list(map(float, x.strip("[]").split())))
                return df_json
        except FileNotFoundError:
            print("Файл не найден")           
        except json.JSONDecodeError:
            print("JSON файл пустой или имеет неправильный формат")

    def match_bad(self, term, list_words):  # вспомог метод
        flag=True
        for word in list_words:
            if word.startswith('_') and term.endswith(word[1 :]):
                flag=False
            elif word.endswith('_') and term.startswith(word[: -1]):
                flag=False
            elif word.endswith('_') and word.startswith('_') and (word[1 : -1] in term):
                flag=False
            elif term==word:
                flag=False
        return flag

    def r_csv(self, data):
        try:
            with open(self.file_csv, "r", encoding="utf-8") as f:  
                df_csv = pd.read_csv(f)
                list_words=df_csv["bad_words"]
                new_data = data[data["term"].apply(lambda x: self.match_bad(x, list_words))].reset_index() 
                new_data.drop(columns=["index"], inplace=True)
                return new_data
        except FileNotFoundError:
            print("Файл не найден") 
        except pd.errors.EmptyDataError:
            print("CSV файл пустой")  
        except pd.errors.ParserError:
            print("Ошибка формата CSV")


    def umap_m(self, data):
        new_data=data.copy()
        embedding = umap.UMAP().fit_transform(new_data['vector'].tolist())
        new_data["metric_one"] = embedding[:, 0]
        new_data["metric_two"] = embedding[:, 1]
        return new_data

    def statistic(self, data):
        new_data=data.copy()
        # new_data.drop(columns='vector', inplace=True)
        new_data["Rank_metric_one"]=data["metric_one"].rank()
        new_data["Rank_metric_two"]=data["metric_two"].rank()
        return new_data
    
    def cluster(self, data): 
        new_data=data.copy()
        kmeans = KMeans(n_clusters=3, random_state=42)  #кол-во кластеров было подобрано с помощью "метода локтя", в точке 3 на графике ошибка перестает существенно уменьшаться
        new_data["cluster"] = kmeans.fit_predict(new_data[["metric_one", "metric_two"]])
        return new_data
   

    def cluster_plt(self, data): 
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='metric_one', y='metric_two', hue='cluster')
        first_word=data.groupby("cluster").agg({"metric_one": "first", "metric_two": "first", "term": "first"})
        for i in range (len(first_word)):
            plt.text(first_word['metric_one'].loc[i], first_word['metric_two'].loc[i], first_word['term'].loc[i], color='r')
        return fig

    def metric_plt(self, data):
        fig, ax  = plt.subplots() 
        new_data=data.melt(id_vars="cluster", value_vars=["metric_one","metric_two"], var_name="metric", value_name="value")
        sns.boxplot(x='cluster', y='value', data=new_data, hue='metric', palette=["#d7c1d6", "#7E739C"], width=0.3)
        count_cl=new_data['cluster'].nunique()
        for i in range(count_cl):
            plt.axvline(i-0.5, color='black', linewidth=1)
        return fig
    
    def save_files(self, data1, data2, data3, data4, fig1, fig2):
        if os.path.exists('export')==False:
            os.mkdir("export")
        data1.to_csv('export/json_data.csv')
        data2.to_csv('export/filter_data.csv')
        data3.to_csv('export/statistic_data.csv')
        data4.to_csv('export/cluster_data.csv')
        fig1.savefig('export/cluster_plt.png')
        fig2.savefig('export/metric_plt.png')