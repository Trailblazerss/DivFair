#!/usr/bin/env python3

import numpy as np
import pandas as pd
import math
import os
from datetime import datetime
from tqdm import tqdm, trange
from .Abstract_Reranker import Abstract_Reranker


class OnlineDoubleClipped(Abstract_Reranker):

    
    def __init__(self, config, weights=None):
        """Initialize the Group-Level Online Double Clipped reranker.
        
        Args:
            config: Configuration dictionary from the framework
            weights: Group weights (target proportions)
        """
        super().__init__(config, weights)
        
        # Double Clipped parameters
        self.delta = config.get('delta', 0.5)
        self.alpha = config.get('alpha', 2.0)
        self.gamma = config.get('gamma', 1.0)
        
        # Online optimization parameters
        self.learning_rate = config.get('learning_rate', 0.1)
        self.momentum_alpha = config.get('momentum_alpha', 1.0)
        self.tau = config.get('tau', 1.0)
        
        # Compute λ based on smoothness condition
        self.lambda_param = self._compute_lambda()
        
    def _compute_lambda(self):
        """Compute λ based on the smoothness condition at v = δ."""
        return np.exp(-1) * (self.delta ** (self.alpha - 1)) / self.gamma
    
    def _objective_derivative(self, v):
        """Compute the derivative of the Double Clipped objective function."""
        if isinstance(v, np.ndarray):
            result = np.zeros_like(v)
            mask_low = v < self.delta
            mask_high = v > self.delta
            
            # For v < δ
            result[mask_low] = (1.0 / self.lambda_param) * np.exp(-v[mask_low] / self.delta) / self.delta
            
            # For v > δ
            if np.abs(self.alpha - 1.0) < 1e-10:  # α = 1
                result[mask_high] = self.gamma / np.maximum(v[mask_high], 1e-10)
            else:  # α ≠ 1
                result[mask_high] = self.gamma * (np.maximum(v[mask_high], 1e-10) ** (-self.alpha))
            
            # At v = δ
            mask_equal = np.isclose(v, self.delta)
            result[mask_equal] = (1.0 / self.lambda_param) * np.exp(-1) / self.delta
            
            return result
        else:
            if v < self.delta:
                return (1.0 / self.lambda_param) * np.exp(-v / self.delta) / self.delta
            elif v > self.delta:
                v_safe = max(v, 1e-10)
                if np.abs(self.alpha - 1.0) < 1e-10:  # α = 1
                    return self.gamma / v_safe
                else:  # α ≠ 1
                    return self.gamma * (v_safe ** (-self.alpha))
            else:  # v == δ
                return (1.0 / self.lambda_param) * np.exp(-1) / self.delta
    

    def _solve_group_constraint_optimization(self, theta_g):
        """Solve constraint optimization for group utilities."""
        # Group utilities: v_g = argmax_{0 ≤ v_g ≤ min(theta_g, 1)} Σ_g mu_g * v_g
        # For framework compatibility, we use the simple solution
        v_g = np.minimum(theta_g, self.tau)
        return v_g
    
    def scale_ranking_scores(self, ranking_score):
        """
        Scale ranking score matrix to [0, 1] range and compute column sums.
        
        Args:
            ranking_score: numpy array of shape [user_size, item_num] with ranking scores
            
        Returns:
            tuple: (scaled_matrix, column_sums)
                - scaled_matrix: ranking scores scaled to [0, 1] per user
                - column_sums: sum of scaled scores for each item across all users
        """
        user_size, item_num = ranking_score.shape
        scaled_matrix = np.zeros_like(ranking_score)
        ranking_score[ranking_score == -1000.0] = 0.0
        # print(f"Scaling ranking score matrix: {user_size} users x {item_num} items")
        
        # Scale each user's scores to [0, 1] independently
        for t in range(user_size):
            w_t_raw = ranking_score[t]
            w_min = np.min(w_t_raw)
            w_max = np.max(w_t_raw)
            
            if w_max > w_min:  # Avoid division by zero
                scaled_matrix[t] = (w_t_raw - w_min) / (w_max - w_min)
            else:
                scaled_matrix[t] = np.ones_like(w_t_raw) * 0.5  # If all scores are the same
        
        # Compute column sums (total scaled score per item)
        column_sums = np.sum(scaled_matrix, axis=0)
        
        return scaled_matrix, column_sums

    def rerank(self, ranking_score, k):
        """
        Group-Level Online Double Clipped reranking.
        
        Args:
            ranking_score: numpy array of shape [user_size, item_num] with ranking scores
            k: number of items to recommend per user
            
        Returns:
            rerank_list: list of reranked item indices for each user
        """
        user_size, item_num = ranking_score.shape
        T = user_size  # Total number of rounds
        delta0 = self.delta
        self.delta = 1.0 / self.group_num * delta0
        self.lambda_param = self._compute_lambda()
        
        # Initialize algorithm state
        mu_g = np.zeros(self.group_num)  # Dual prices per group
        theta_g = np.zeros(self.group_num)  # Phased budget per group
        v_g = np.zeros(self.group_num)  # Group utilities
        g_prev = np.zeros(self.group_num)  # Previous momentum gradient
        
        # Initialize tracking for evolution analysis
        evolution_data = []
        
        rerank_list = []
        # # Scale ranking scores to [0, 1] to avoid numerical issues while preserving preferences
        ranking_score, column_sums = self.scale_ranking_scores(ranking_score)

        # Online algorithm: process users sequentially
        for t in trange(user_size, desc="Online Double Clipped"):
            w_t = ranking_score[t]  # Current user's preference scores
            
            # 1. Compute objective gradients for groups
            grad_f_g = self._objective_derivative(v_g)
            
            # Switching penalty: m_i = 0 if Σ_g M[i,g] * theta_g[g] <= tau else -∞
            # item_theta = np.matmul(self.M, theta_g)
            m_i = np.where(theta_g <= self.tau, 0.0, -1e10)
            
            # Score: q_i = w_ti * ((1/T) * item_group_grad[i] - item_group_dual[i]) + m_i
            q_i = w_t * grad_f_g - mu_g + m_i
            
            # 3. Select top-K items (this satisfies Σ_i x_ti = k constraint)
            top_k_indices = np.argsort(q_i)[-k:]
            
            # Sort selected items by original relevance score (descending) for framework compatibility
            selected_scores = w_t[top_k_indices]
            sorted_indices = np.argsort(selected_scores)[::-1]
            final_items = top_k_indices[sorted_indices]
            rerank_list.append(final_items)

            # 4. Compute item allocations directly
            # allocation per item: 1 if item is selected, 0 otherwise
            allocation_g = np.zeros(item_num)
            allocation_g[final_items] = 1.0
            normalized_allocation = allocation_g / k  # Normalize by number of selected items

            # 5. Update group budget and utilities
            # Update phased group budget: theta_g = theta_g + (1/T) * (allocation_g / k)
            theta_g = theta_g + (1.0 / T) * normalized_allocation
            
            # Solve constraint optimization: v_g = argmax_{0 ≤ v_g ≤ min(theta_g, 1)} Σ_g mu_g * v_g
            v_g = self._solve_group_constraint_optimization(theta_g)
            
            # 6. Group-level dual update with momentum
            # Constraint subgradient: g_g^tilde = group_weights - (allocation_g / k)
            g_g_tilde = self.weights / self.group_num - normalized_allocation
            g_g = self.momentum_alpha * g_g_tilde + (1 - self.momentum_alpha) * g_prev
            g_prev = g_g
            
            # Update dual variables: mu_g_{t+1} = max(0, mu_g_t - eta_t * g_g)
            eta_t = self.learning_rate / math.sqrt(item_num)
            mu_g = np.maximum(0.0, mu_g - eta_t * g_g) 
           
        
        print(f"Optimization completed!")
        
        # Calculate final fairness metrics for display
        nonzero_groups = v_g[v_g > 1e-8]
        self.delta = delta0
        if len(nonzero_groups) > 0:
            utility_minmax = np.min(nonzero_groups) / (np.max(nonzero_groups) + 1e-8)
            print(f"Final min-max group utility ratio: {utility_minmax:.4f}")
        
        return rerank_list
