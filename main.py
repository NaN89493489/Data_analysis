from pipeline import AnalysisData

def main():
    try:
        test=AnalysisData("data.json", "bad_words.csv")
        data_json=test.r_json()
        filter_data=test.r_csv(data_json)
        umap_data=test.umap_m(filter_data)
        statistic_data=test.statistic(umap_data)
        cluster_data=test.cluster(umap_data)
        fig1 = test.cluster_plt(cluster_data)
        fig2 = test.metric_plt(cluster_data)
        test.save_files(data_json, filter_data, statistic_data, cluster_data, fig1, fig2)
    except Exception as e:
        print("Ошибка:", e)

if __name__ == "__main__":
    main()
