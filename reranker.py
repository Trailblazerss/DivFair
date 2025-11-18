import numpy as np
import os
import yaml
from scipy.sparse import save_npz, load_npz
from DivFair import DivFair
from metric import dcg, MMF, Gini, Entropy, MinMaxRatio, EF, Coverage, WeightedCoverage
from datetime import datetime
import json

from utils import Build_Adjecent_Matrix


class RecReRanker(object):
    def __init__(self, train_config):
        """Initialize In-processing and base models.

        :param train_config: Your custom config files.
        """


        self.dataset = train_config['dataset']
        #self.stage = stage
        self.train_config = train_config


    def load_configs(self, dir):
        """
           Loads and merges configuration files for the model, dataset, and evaluation.

           This function loads multiple YAML configuration files, including the process configuration,
           dataset-specific settings, model configurations, and evaluation parameters. All configurations
           are merged, with the highest priority given to the class's own `config` attribute.

           :param dir: The directory where the main process configuration file is located.
           :return: A dictionary containing the merged configuration from all files.
        """
        print("start to load config...")
        with open(os.path.join("processed_dataset", self.dataset, "process_config.yaml"), 'r') as f:
            config = yaml.safe_load(f)
        # print(train_data_df.head())

        if self.train_config['fair-rank'] == True:
            print("start to load model...")
            with open(os.path.join("properties", "models.yaml"), 'r') as f:
                model_config = yaml.safe_load(f)

            model_path = os.path.join("properties", "models", self.train_config['model'] + ".yaml")
            # if not os.path.exists(model_path):
            #     raise NotImplementedError("we do not support such model type!")
            with open(model_path, 'r') as f:
                model_config.update(yaml.safe_load(f))
            config.update(model_config)

        with open(os.path.join("properties", "evaluation.yaml"), 'r') as f:
            config.update(yaml.safe_load(f))

        config.update(self.train_config)  ###train_config has highest rights
        print("your loading config is:")
        print(config)

        return config

    def rerank(self):
        """
            Training post-processing main workflow.
        """

        dir = os.path.join("processed_dataset", self.dataset)
        config = self.load_configs(dir)

        ranking_score_path = os.path.join("log", config['ranking_store_path'])
        if not os.path.exists(ranking_score_path):
            raise ValueError(f"do not exist the path {ranking_score_path}, please check the path or run the ranking phase to generate scores for re-ranking !")
        print("loading ranking scores....")
        file = os.path.join(ranking_score_path, "ranking_scores.npz")
        ranking_scores = load_npz(file).toarray() #[user_num, item_num]

        user_num, item_num = ranking_scores.shape
        scaled_matrix = np.zeros_like(ranking_scores)
        ranking_scores[ranking_scores == -1000.0] = 0.0
        # Scale each user's scores to [0, 1]
        for t in range(user_num):
            w_t_raw = ranking_scores[t]
            w_min = np.min(w_t_raw)
            w_max = np.max(w_t_raw)
            if w_max > w_min:  # Avoid division by zero
                scaled_matrix[t] = (w_t_raw - w_min) / (w_max - w_min)
            else:
                scaled_matrix[t] = np.ones_like(w_t_raw) * 0.5  # If all scores are the same
        ranking_scores = scaled_matrix

        if config['fair-rank'] == False:
            config['model'] = None
        if config['model'] == 'DivFair':
            Reranker = DivFair(config)
        elif config['model'] == None:
            print("No model loaded!")
        else:
            raise NotImplementedError(f"We do not support the model type {self.train_config['model']}")

        metrics = ["ndcg", "u_loss"]
        rerank_result = {}
        exposure_result = {}
        for k in config['topk']:
            rerank_result.update({f"{m}@{k}":0 for m in metrics})

            if config['fair-rank'] == True:
                rerank_list = Reranker.rerank(ranking_scores, k)
            else:
                result_item = np.argsort(ranking_scores,axis=-1)[:,::-1]
                rerank_list = result_item[:,:k]

            exposure_list = np.zeros(config['group_num'])
            for u in range(len(rerank_list)):
                sorted_result_score = np.sort(ranking_scores[u])[::-1]
                true_dcg = dcg(sorted_result_score, k)
                rerank_items = rerank_list[u]

                for i in rerank_items:
                    if config['fairness_type'] == "Exposure":
                        exposure_list[i] += 1
                    else:
                        exposure_list[i] += np.round(ranking_scores[u][i], config['decimals'])
                reranked_score = ranking_scores[u][rerank_items]
                pre_dcg = dcg(np.sort(reranked_score)[::-1], k)
                rerank_result[f"ndcg@{k}"] += pre_dcg/true_dcg
                rerank_result[f"u_loss@{k}"] += (np.sum(sorted_result_score[:k]) - np.sum(reranked_score[:k]))/k
            
            rerank_result[f"ndcg@{k}"] /= len(rerank_list)
            rerank_result[f"u_loss@{k}"] /= len(rerank_list)

            for fairness_metric in self.train_config['fairness_metrics']:
                if fairness_metric == 'MinMaxRatio':
                    rerank_result[f"MinMaxRatio@{k}"] = MinMaxRatio(exposure_list)
                elif fairness_metric == 'MMF':
                    rerank_result[f"MMF@{k}"] = MMF(exposure_list)
                elif fairness_metric == 'Entropy':
                    rerank_result[f"Entropy@{k}"] = Entropy(exposure_list)
                elif fairness_metric == 'GINI':
                    rerank_result[f"GINI@{k}"] = Gini(exposure_list)
                elif fairness_metric == 'EF':
                    rerank_result[f"EF@{k}"] = EF(exposure_list)
                elif fairness_metric == 'Coverage':
                    rerank_result[f"Catalog_Coverage@{k}"] = Coverage(exposure_list)
                elif fairness_metric == 'Coverage_with_threshold':
                    rerank_result[f"Catalog_Coverage_with_threshold@{k}"] = WeightedCoverage(exposure_list, threshold = 0.5 * np.mean(exposure_list))

            exposure_result[f"top@{k}"] = str(list(exposure_list))


        for k in rerank_result.keys():
            rerank_result[k] = np.round(rerank_result[k], config['decimals'])


        today = datetime.today()
        today_str = f"{today.year}-{today.month}-{today.day}"
        log_dir = os.path.join("log", f"{today_str}_{config['log_name']}")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(os.path.join(log_dir, 'test_result.json'), 'w') as file:
            json.dump(rerank_result, file)
        with open(os.path.join(log_dir, 'exposure_result.json'), 'w') as file:
            json.dump(exposure_result, file)
        print(rerank_result)

        with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
            yaml.dump(config, f)

        print(f"result and config dump in {log_dir}")






