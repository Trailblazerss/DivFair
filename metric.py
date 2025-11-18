from sklearn.metrics import roc_auc_score
import numpy as np

##############ranking metrics #############################

def AUC_score(y_scores, y_true):
    r"""AUC_ (also known as Area Under Curve) is used to evaluate the two-class model, referring to
        the area under the ROC curve.

        .. _AUC: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

        Note:
            This metric does not calculate group-based AUC which considers the AUC scores
            averaged across users. It is also not limited to k. Instead, it calculates the
            scores on the entire prediction results regardless the users. We call the interface
            in `scikit-learn`, and code calculates the metric using the variation of following formula.

        .. math::
            \mathrm {AUC} = \frac {{{M} \times {(N+1)} - \frac{M \times (M+1)}{2}} -
            \sum\limits_{i=1}^{M} rank_{i}} {{M} \times {(N - M)}}

        :math:`M` denotes the number of positive items.
        :math:`N` denotes the total number of user-item interactions.
        :math:`rank_i` denotes the descending rank of the i-th positive item.
    """
    auc_score = roc_auc_score(y_true, y_scores)
    return auc_score

def dcg(scores, k):
    """
    Calculate the Discounted Cumulative Gain (DCG) at rank k.
    """
    scores = np.array(scores)[:k]
    if scores.size:
        return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
    return 0.0


def NDCG(ranking_list, label_list, k):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality,
    where positions are discounted logarithmically. It accounts for the position of the hit by assigning
    higher scores to hits at top ranks.

    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \mathrm {NDCG@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{\sum_{i=1}^{\min (|R(u)|, K)}
        \frac{1}{\log _{2}(i+1)}} \sum_{i=1}^{K} \delta(i \in R(u)) \frac{1}{\log _{2}(i+1)})

    :math:`\delta(·)` is an indicator function.
    """
    sorted_label = np.sort(label_list)[::-1]
    label_dcg = dcg(sorted_label, k)
    ranking_dcg = dcg(ranking_list, k)
    ndcg = ranking_dcg/label_dcg

    return ndcg

def HR(ranking_list, label_list, k):
    r"""HR_ (also known as truncated Hit-Ratio) is a way of calculating how many 'hits'
        you have in an n-sized list of ranked items. If there is at least one item that falls in the ground-truth set,
        we call it a hit.

        .. _HR: https://medium.com/@rishabhbhatia315/recommendation-system-evaluation-metrics-3f6739288870

        .. math::
            \mathrm {HR@K} = \frac{1}{|U|}\sum_{u \in U} \delta(\hat{R}(u) \cap R(u) \neq \emptyset),

        :math:`\delta(·)` is an indicator function. :math:`\delta(b)` = 1 if :math:`b` is true and 0 otherwise.
        :math:`\emptyset` denotes the empty set.
    """
    sorted_label = np.sort(label_list)[::-1]
    hr = np.sum(ranking_list[:k])/np.sum(sorted_label[:k])
    return hr


def MRR(ranking_list, k):
    r"""The MRR_ (also known as Mean Reciprocal Rank) computes the reciprocal rank
        of the first relevant item found by an algorithm.

        .. _MRR: https://en.wikipedia.org/wiki/Mean_reciprocal_rank

        .. math::
           \mathrm {MRR@K} = \frac{1}{|U|}\sum_{u \in U} \frac{1}{\operatorname{rank}_{u}^{*}}

        :math:`{rank}_{u}^{*}` is the rank position of the first relevant item found by an algorithm for a user :math:`u`.
    """
    mrr = 0
    for index, i in enumerate(ranking_list[:k]):
        if i > 0:
            mrr = i/(index+1)
            break
    return mrr


##################fairness metric#######################

def reconstruct_utility(utility_list, weights, group_mask):
    """
        Reconstruct utility by re-weighting them and masking the utility of certain unused groups.
        :param utility_list: array for item/user utilities
        :param weights: array for item/user utilities weights
        :param group_mask: bool array for whether computed the group utilityes
        :return: re-constructed utility array
    """

    if not weights:
        weights = np.ones_like(utility_list)

    utility_list = np.array(utility_list)
    weights = np.array(weights)
    weighted_utility = utility_list * weights

    if group_mask:
        weighted_utility = mask_utility(weighted_utility, group_mask)


    return np.array(weighted_utility)

def mask_utility(utility, group_mask):
    """
    Mask the utility values based on the provided group mask.

    This function filters out the utility values where the corresponding
    group mask element is zero, effectively removing them from the output.


    :param utility: array for item/user utilities
    :param group_mask: bool array for whether computed the group utilityes
    :return: masked utility array
    """


    masked_utility = []
    for i, m in enumerate(group_mask):
        if m == 0:
            masked_utility.append(utility[i])

    return np.array(masked_utility)

def MMF(utility_list, ratio=0.5, weights=None, group_mask = None):
    """
    Calculate the Max-min Fairness (MMF) index based on a given utility list.

    Parameters
    :param utility: array-like
        A list or array representing the utilities of resources or users.
    :param ratio: float, optional
        The fraction of the minimum utilities to consider for the MMF calculation. Defaults to 0.5.
    :param ratio: float, optional
        The fraction of the minimum utilities to consider for the MMF calculation. Defaults to 0.5.
    :param ratio: float, optional
        The fraction of the minimum utilities to consider for the MMF calculation. Defaults to 0.5.
    :param weights : array-like, optional
        An optional list or array of weights corresponding to each utility in `utility_list`.
        If provided, utilities are multiplied by their respective weights before sorting.
        Defaults to None, implying equal weighting.
    :param group_mask : array-like, optional
        An optional list or array used to selectively apply weights. If provided, it must have the same length as
        `utility_list` and `weights`. Defaults to None, indicating no group-based weighting.

    :return: The computed MMF index, representing the fairness of the allocation.
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)

    MMF_length = int(ratio * len(utility_list))
    utility_sort = np.sort(weighted_utility)

    mmf = np.sum(utility_sort[:MMF_length])/np.sum(weighted_utility)

    return mmf

def MinMaxRatio(utility_list, weights=None, group_mask = None):
    """
    This function computes the minimum-to-maximum ratio of a list of utilities, optionally weighted and grouped.

    :param utility_list (list of float): A list containing numerical utility values.
    :param weights (list of float, optional): A list of weights corresponding to the utilities in utility_list. If provided,
      each utility is multiplied by its respective weight. If None, all utilities are considered with equal weight.
    :param group_mask (list of int or bool, optional): A mask indicating groups within the utility_list. If provided, it must
      be of the same length as utility_list. Groups are defined by consecutive True or 1 values. If None, no grouping is applied.

    :return: float: The computed minimum-to-maximum ratio of the (weighted) utilities.
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)
    return np.min(weighted_utility) / np.max(weighted_utility)


def Gini(utility_list, weights=None, group_mask = None):
    """
    This function computes the Gini coefficient, a measure of statistical dispersion intended to represent income inequality within a nation or social group.
    The Gini coefficient is calculated based on the cumulative distribution of values in `utility_list`, which can optionally be weighted and masked.

    :param utility_list: array_like
        A 1D array representing individual utilities. The utilities are used to compute the Gini coefficient.
    :param weights: array_like, optional
        A 1D array of weights corresponding to `utility_list`. If provided, each utility value is multiplied by its respective weight before calculating the Gini coefficient. Defaults to None, implying equal weighting.
    :param group_mask: array_like, optional
        A 1D boolean array used to selectively include elements from `utility_list`. If provided, only the elements where the mask is True are considered in the calculation. Defaults to None, meaning all elements are included.

    :return: float: The computed Gini coefficient, ranging from 0 (perfect equality) to 1 (maximal inequality).
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)

    values = np.sort(weighted_utility)
    n = len(values)

    # gini compute
    cumulative_sum = np.cumsum(values)
    gini = (n + 1 - 2 * (np.sum(cumulative_sum) / cumulative_sum[-1])) / n

    return gini

def Entropy(utility_list, weights=None, group_mask = None):
    """
    Calculate the entropy of a distribution given by `utility_list`, optionally
    weighted by `weights` and filtered by `group_mask`. Entropy measures the
    disorder or uncertainty in the distribution.

    :param utility_list: list or array-like
        A list or array representing utility values for each item.
    :param weights: list or array-like, optional
        A list or array of weights corresponding to each utility value. If not provided,
        all utilities are considered equally weighted. Defaults to None.
    :param group_mask : list or array-like, optional
        A boolean mask indicating which utilities to include in the calculation.
        If not provided, all utilities are included. Defaults to None.

    :return: float: The calculated entropy of the (potentially weighted and masked) distribution.

    Notes
    - Entropy is calculated as H = -sum(p * log2(p)), where p is the probability of each event.
    - Probabilities are normalized to ensure their sum equals 1.
    - To avoid taking the log of zero, a small constant (1e-9) is added to each probability before calculating the entropy.
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)

    values = np.array(weighted_utility)
    values = values / np.sum(values)

    # H = - sum(p * log2(p))
    # avoid 0 case
    entropy_value = -np.sum(values * np.log2(values + 1e-9))
    return entropy_value

def ElasticFair(utils,t):
    """
       Calculate the elasticfair metric from the paper of SIGIR 2025:
       Understanding Accuracy-Fairness Trade-offs in Re-ranking through Elasticity in Economics


       :param utils: list or array-like
           A list or array representing utility values for each item.
       :param t: taxation rate
           different t will align with different fairness metric
           for example t=0: Entropy, t=1: Nash fairness, t=\infty: max-min fairness

       :return: float: The fairness metric value under specific t.
    """

    if t != 0:
        sign = np.sign(1-t)
        utils_g = np.sum(np.power(utils,1-t))
        utils_g = np.power(utils_g, 1/t)
        return sign * utils_g
    else:
        entropy = - np.sum(utils * np.log(utils))
        return np.exp(entropy)

def EF(utility_list, M = 50, weights=None, group_mask = None):
    """
       Calculate the EF metric from the paper of SIGIR 2025:
       Understanding Accuracy-Fairness Trade-offs in Re-ranking through Elasticity in Economics


       :param utils: list or array-like
           A list or array representing utility values for each item.
       :param M: bound of the integral

       :return: float: The fairness metric EF value.
    """

    utility_list = reconstruct_utility(utility_list, weights, group_mask)
    utility_list = utility_list / np.sum(utility_list)

    t = np.linspace(1 - M, 1 + M, 200)
    fair = []
    for i in t:
        fair.append(ElasticFair(utility_list, i))
    integral = np.trapz(fair, t)
    return integral/(2*M*len(utility_list))

##############diversity metrics #############################

def Coverage(utility_list, weights=None, group_mask=None, normalize=True):
    """
    Calculate the coverage diversity metric based on the number of distinct groups 
    represented in the recommendation set.
    
    Coverage measures how many different categories/groups are represented in the 
    recommendations, indicating the diversity of the recommendation set. Higher 
    coverage means more diverse recommendations across different groups.
    
    :param utility_list: array-like
        A list or array representing the utilities of groups. Non-zero values 
        indicate that the group is represented in the recommendations.
    :param weights: array-like, optional
        A list or array of weights corresponding to each group utility. 
        If provided, weighted coverage considers the relative importance of groups.
        Defaults to None, implying equal weighting.
    :param group_mask: array-like, optional
        A boolean mask indicating which groups to include in the coverage calculation.
        If provided, only groups where the mask is True are considered.
        Defaults to None, meaning all groups are included.
    :param normalize: bool, optional
        Whether to normalize coverage by the total number of available groups.
        If True, returns coverage as a fraction [0, 1]. If False, returns the 
        absolute number of covered groups. Defaults to True.
        
    :return: float
        The coverage score. If normalize=True, returns a value in [0, 1] where 
        1 means all groups are covered. If normalize=False, returns the count 
        of covered groups.
        
    Examples:
        >>> # Example with 5 groups, 3 are represented
        >>> utilities = [0.2, 0.0, 0.3, 0.1, 0.0]  # Groups 0, 2, 3 covered
        >>> Coverage(utilities)  # Returns 0.6 (3/5 groups covered)
        
        >>> # Example with weights (some groups more important)
        >>> utilities = [0.2, 0.0, 0.3, 0.1, 0.0]
        >>> weights = [2.0, 1.0, 1.0, 1.0, 3.0]
        >>> Coverage(utilities, weights=weights)  # Weighted coverage
        
        >>> # Example with group mask (only consider subset of groups)
        >>> utilities = [0.2, 0.0, 0.3, 0.1, 0.0]
        >>> mask = [True, True, True, False, False]  # Only consider first 3 groups
        >>> Coverage(utilities, group_mask=mask)  # Returns 0.67 (2/3 groups covered)
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)
    
    # Count groups with non-zero utility (i.e., groups that are covered)
    covered_groups = np.count_nonzero(weighted_utility)
    total_groups = len(weighted_utility)
    
    if normalize:
        # Return normalized coverage [0, 1]
        return covered_groups / total_groups if total_groups > 0 else 0.0
    else:
        # Return absolute count of covered groups
        return covered_groups


def WeightedCoverage(utility_list, weights=None, group_mask=None, threshold=0.01):
    """
    Calculate weighted coverage that considers both the presence and magnitude 
    of group utilities in recommendations.
    
    Unlike basic coverage which only counts whether groups are represented,
    weighted coverage considers how well each group is represented based on
    the magnitude of their utilities. This provides a more nuanced view of
    diversity that accounts for the strength of representation.
    
    :param utility_list: array-like
        A list or array representing the utilities of groups.
    :param weights: array-like, optional
        A list or array of weights corresponding to each group utility.
        Defaults to None, implying equal weighting.
    :param group_mask: array-like, optional
        A boolean mask indicating which groups to include in the calculation.
        Defaults to None, meaning all groups are included.
    :param threshold: float, optional
        Minimum utility threshold for a group to be considered "covered".
        Groups with utility below this threshold are not counted as covered.
        Defaults to 0.01.
        
    :return: float
        The weighted coverage score in [0, 1], where 1 indicates perfect 
        weighted coverage across all groups.
        
    Examples:
        >>> # Groups with different levels of representation
        >>> utilities = [0.4, 0.3, 0.2, 0.1, 0.0]  # Decreasing representation
        >>> WeightedCoverage(utilities)  # Considers magnitude of representation
        
        >>> # With threshold - only well-represented groups count
        >>> utilities = [0.4, 0.05, 0.3, 0.02, 0.0]
        >>> WeightedCoverage(utilities, threshold=0.1)  # Only groups with >0.1 count
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)
    
    # Apply threshold filter
    above_threshold = weighted_utility >= threshold
    filtered_utility = weighted_utility * above_threshold

    covered_groups = np.count_nonzero(filtered_utility)
    total_groups = len(weighted_utility)
    
    # Return normalized coverage [0, 1]
    return covered_groups / total_groups if total_groups > 0 else 0.0
    
    # # Calculate weighted coverage as the effective coverage
    # # Uses Shannon diversity-inspired approach
    # if np.sum(filtered_utility) == 0:
    #     return 0.0
    
    # # Normalize utilities to probabilities
    # probabilities = filtered_utility / np.sum(filtered_utility)
    
    # # Calculate effective number of groups (exponential of Shannon entropy)
    # # This gives more weight to even distribution across groups
    # entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))
    # effective_groups = np.exp(entropy)
    
    # # Normalize by theoretical maximum (total number of groups)
    # max_groups = len(weighted_utility)
    # return effective_groups / max_groups if max_groups > 0 else 0.0


def IntraListDiversity(utility_list, weights=None, group_mask=None, method='gini'):
    """
    Calculate intra-list diversity (ILD) based on the distribution of group utilities
    in recommendations.
    
    Intra-list diversity measures how evenly distributed the utilities are across 
    different groups. Higher ILD indicates more diverse recommendations where utilities
    are more evenly spread across groups rather than concentrated in a few groups.
    
    :param utility_list: array-like
        A list or array representing the utilities of groups. Higher values 
        indicate that the group is more represented in the recommendations.
    :param weights: array-like, optional
        A list or array of weights corresponding to each group utility.
        Defaults to None, implying equal weighting.
    :param group_mask: array-like, optional
        A boolean mask indicating which groups to include in the calculation.
        Defaults to None, meaning all groups are included.
    :param method: str, optional
        Method to calculate diversity:
        - 'gini': Uses 1 - Gini coefficient (more even = higher diversity)
        - 'entropy': Uses normalized entropy (higher entropy = higher diversity)  
        - 'variance': Uses 1 - normalized variance (lower variance = higher diversity)
        Defaults to 'gini'.
        
    :return: float
        The intra-list diversity score in [0, 1], where 1 indicates maximum 
        diversity (utilities evenly distributed across all groups).
        
    Examples:
        >>> # Even distribution across groups (high diversity)
        >>> utilities = [0.25, 0.25, 0.25, 0.25]  # Equal utilities
        >>> IntraListDiversity(utilities)  # Returns high diversity (~1.0)
        
        >>> # Concentrated in one group (low diversity)
        >>> utilities = [1.0, 0.0, 0.0, 0.0]  # All utility in first group
        >>> IntraListDiversity(utilities)  # Returns low diversity (~0.0)
        
        >>> # Mixed distribution (medium diversity)
        >>> utilities = [0.6, 0.3, 0.1, 0.0]  # Uneven but not completely concentrated
        >>> IntraListDiversity(utilities)  # Returns medium diversity
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)
    
    # Handle edge cases
    if len(weighted_utility) <= 1:
        return 0.0  # No diversity with 0 or 1 groups
    
    # Normalize utilities to probabilities
    total_utility = np.sum(weighted_utility)
    if total_utility == 0:
        return 0.0  # No utility means no diversity
    
    probabilities = weighted_utility / total_utility
    
    if method == 'gini':
        # Use 1 - Gini coefficient (Gini measures inequality, so 1-Gini measures equality/diversity)
        gini_coeff = Gini(weighted_utility)
        return 1.0 - gini_coeff
        
    elif method == 'entropy':
        # Use normalized entropy (higher entropy = more diversity)
        entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        max_entropy = np.log2(len(probabilities))  # Maximum possible entropy
        return entropy_value / max_entropy if max_entropy > 0 else 0.0
        
    elif method == 'variance':
        # Use 1 - normalized variance (lower variance = higher diversity)
        # For uniform distribution, variance should be 0, giving diversity = 1
        uniform_prob = 1.0 / len(probabilities)
        variance = np.var(probabilities)
        max_variance = uniform_prob * (1 - uniform_prob)  # Maximum possible variance
        normalized_variance = variance / max_variance if max_variance > 0 else 0.0
        return 1.0 - normalized_variance
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'gini', 'entropy', or 'variance'.")


