#!/usr/bin/env python3

import numpy as np
import pandas as pd
import math
import os
from datetime import datetime
from tqdm import tqdm, trange
from Abstract_Reranker import Abstract_Reranker


class DivFair(Abstract_Reranker):

    
    def __init__(self, config, weights=None):
        """Initialize the reranker.
        
        Args:
            config: Configuration dictionary from the framework
            weights: weights, set equal for every item
        """
        super().__init__(config, weights)
        
        # Model parameters
        self.delta = config.get('delta', 0.5)
        self.alpha = config.get('alpha', 2.0)
        self.gamma = config.get('gamma', 1.0)
        
        # Optimization parameters
        self.learning_rate = config.get('learning_rate', 0.1)
        self.momentum_alpha = config.get('momentum_alpha', 1.0)
        self.tau = config.get('tau', 1.0)
        
        # Compute λ based on smoothness condition
        self.lambda_param = self._compute_lambda()
        
    def _compute_lambda(self):
        """Compute λ based on the smoothness condition at v = δ."""
        return np.exp(-1) * (self.delta ** (self.alpha - 1)) / self.gamma
    
    def _objective_derivative(self, v):
        """Compute the derivative of the objective function."""
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
        v_g = np.minimum(theta_g, self.tau)
        return v_g
    
    def rerank(self, ranking_score, k):
        """
        Reranking
        
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

        print(f"k = {k}")
        
        # Initialize algorithm state
        mu_g = np.zeros(self.group_num)  # Dual prices per group
        theta_g = np.zeros(self.group_num)  # Phased budget per group
        v_g = np.zeros(self.group_num)  # Group utilities
        g_prev = np.zeros(self.group_num)  # Previous momentum gradient
        
        rerank_list = []

        # Online algorithm: process users sequentially
        for t in range(user_size):
            w_t = ranking_score[t]  # Current user's preference scores
            
            # Compute objective gradients for groups
            grad_f_g = self._objective_derivative(v_g)
            # Switching penalty: m_i = 0 if Σ_g M[i,g] * theta_g[g] <= tau, else infinity
            m_i = np.where(theta_g <= self.tau, 0.0, -1e10)
            q_i = w_t * grad_f_g * 1.0 / T - mu_g + m_i
            # Select top-K items (this satisfies Σ_i x_ti = k constraint)
            top_k_indices = np.argsort(q_i)[-k:]
            # Sort selected items by original relevance score (descending) for framework compatibility
            selected_scores = w_t[top_k_indices]
            sorted_indices = np.argsort(selected_scores)[::-1]
            final_items = top_k_indices[sorted_indices]
            rerank_list.append(final_items)

            # Compute item allocations directly
            # allocation per item: 1 if item is selected, 0 otherwise
            allocation_g = np.zeros(item_num)
            allocation_g[final_items] = 1.0
            normalized_allocation = allocation_g / k  # Normalize by number of selected items

            # Update group budget and utilities
            # Update phased group budget: theta_g = theta_g + (1/T) * (allocation_g / k)
            theta_g = theta_g + (1.0 / T) * normalized_allocation
            # Solve constraint optimization: v_g = argmax_{0 ≤ v_g ≤ min(theta_g, 1)} Σ_g mu_g * v_g
            v_g = self._solve_group_constraint_optimization(theta_g)
            
            # Group-level dual update with momentum
            # Constraint subgradient: g_g^tilde = group_weights - (allocation_g / k)
            g_g_tilde = self.weights / self.group_num - normalized_allocation
            g_g = self.momentum_alpha * g_g_tilde + (1 - self.momentum_alpha) * g_prev
            g_prev = g_g
            
            # Update dual variables: mu_g_{t+1} = max(0, mu_g_t - eta_t * g_g)
            eta_t = self.learning_rate / math.sqrt(item_num)
            mu_g = np.maximum(0.0, mu_g - eta_t * g_g) 
           
        print(f"Optimization completed!")
        # Calculate final fairness metrics for demonstration
        nonzero_groups = v_g[v_g > 1e-8]
        self.delta = delta0
        if len(nonzero_groups) > 0:
            utility_minmax = np.min(nonzero_groups) / (np.max(nonzero_groups) + 1e-8)
            print(f"Final min-max utility ratio: {utility_minmax:.4f}")

        return rerank_list
