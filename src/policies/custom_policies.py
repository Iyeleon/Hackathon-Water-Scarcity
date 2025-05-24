import numpy as np

def outlier_score_mad(data):
    data = np.array(data)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return 1.0  # No spread = no outlier detectable # return 0 with logarith
    modified_z = 0.6745 * (data - median) / mad
    return np.max(np.abs(modified_z))  # The highest "outlier-ness"
    
def adaptive_quota(
        self,
        crisis_level: int,
        actors_priority: np.ndarray,
        avg_pump: np.ndarray,
        DOE: float,
        DCR: float,
        rate: float = 2.266202914546536, #1.5,
        # water_reserve_buffer: float = 1.1616014660103775, #1.0,
        priority_effect: float = 3.0,#10 #2.4338629141770904 #2.35 #2.3 with logarithm
    ) -> np.ndarray:
    """
    Deterministic quota policy based on actor priority, avg_pump, and crisis level.
    Always allocates available water scaled by crisis level and resilience.
    """
    # if in alert or crisis - water usage from last step took a lot of water
    # prioritize quotas such that total water taken based on prediction does not trigger a crisis
    # crisis level will be based on probability of being in crisis in the next phase if all actor demands are satisfied. 
    # all actor demands is based on average pump. 

    # get water predictions, total_Demand
    total_demand = avg_pump.sum()

    # create water reserve buffer to ensure there is enough water during drought
    # available_water = total_demand * rate #- (water_reserve_buffer * (DCR + DOE) / 2) # water available to avert crisis
    available_water = total_demand * rate
    
    # available_water = max(0, available_water)

    if available_water == 0:
        return np.zeros_like(avg_pump)

    # give sufficient water during abundance
    if crisis_level <= -1:
        quota = avg_pump * rate
    else:
        # get median absolute deviation to identify large demand actors
        scale = outlier_score_mad(avg_pump)

        # option 1 - log scale the factors
        # scale = 1 / np.power(scale, 0.25)
        # priority_weights = np.log(actors_priority + 1) + actors_priority ** max(0, crisis_level * priority_effect * scale)

        # option 2 - exp scale - more interpretable
        # allocates available water by priority and crisis_level
        scale = 1 / np.power(scale, 0.25)
        priority_weights = np.exp(actors_priority * crisis_level * priority_effect * scale) * (actors_priority > 0).astype(int)

        # compute weighted allocation 
        weights = avg_pump * priority_weights
        total_weight = np.sum(weights)
        quota = (weights / total_weight) * available_water
        
    # satisfaction = quota / np.array(avg_pump)
    return quota

def hard_ecofair_incentives_policy(
    self,
    actions,
    actors_priority: np.ndarray,
    avg_incomes: np.ndarray,
    water_pump: np.ndarray,
    avg_pump: np.ndarray,
    is_crisis: np.ndarray,
    water_flows: np.ndarray,
    quota: np.ndarray,
    DOE = 15,
    DCR = 10,
    max_subvention_weight = -9.838033313653508, #-1.0,
    max_fine_weight = 0.0516587828927856152, #0.05,
    policy_rate = 2.207608655434449, #10.0,
    policy_skew = 4.887128330883843, #2.0,
    policy_spread = 26.092559856664767, #10.0,
    i_alpha = 0.49724850589238545, #0.6375574713552131, #0.0,
    subsidy_fine_ratio = 0.8776734437380462,#0.9214017645310711, #0.5,
) -> np.ndarray:
    """
    Continuous incentive policy using generalized logistic function
    """
    # overuse or underuse penalty
    # -ve values are subsidies and +values are fines
    # actors using less than quota will be get subsidies
    # actors using above quota will get fined

    # get crisis level from last run
    crisis_level = is_crisis[-1]
    
    # get water predictions, total_demand to estimate remaining water
    estimated_remaining_water = water_flows[-1] - water_pump.sum()

    # use remaining water to check if in crisis after pumping
    # this factors into penalizing actors actions with knowledge of prediction
    if estimated_remaining_water <= DCR:
        next_crisis = 2
    elif estimated_remaining_water <= ((DCR + DOE) / 2):
        next_crisis = 1
    elif estimated_remaining_water <= DOE:
        next_crisis = 0
    else:
        next_crisis = -1

    # new crisis level
    # weights to determine focus on current or past crisis
    crisis_level = i_alpha * next_crisis + (1 - i_alpha) * crisis_level

    # check for ongoing crisis
    if is_crisis[-1] >= 1 and is_crisis[-2] >= 1:
        crisis_level = is_crisis[-1] + is_crisis[-2]

    # round to give integer
    hard_crisis_level = round(crisis_level, 0)

    # compute policy controls
    c = hard_crisis_level + 1
    penalty_spread = policy_spread * np.exp(-policy_skew * c)
    priority_weight = np.exp(2 - actors_priority)
    fairness_regularization = penalty_spread + priority_weight

    # compute overuse
    overuse = water_pump - quota

    # compute penalty rate
    penalty_rate = policy_rate * overuse / (quota + fairness_regularization)

    # diverge rate of subsidy and penalty
    penalty_rate = np.where(penalty_rate < 0, penalty_rate * subsidy_fine_ratio, penalty_rate)

    # clip rates to max fine and subsidies
    penalty_rate = np.clip(penalty_rate, max_subvention_weight, max_fine_weight)

    # compute subventions and fines
    incentives = avg_incomes * penalty_rate

    return incentives

def smooth_ecofair_incentives_policy(
    self,
    actions,
    actors_priority: np.ndarray,
    avg_incomes: np.ndarray,
    water_pump: np.ndarray,
    avg_pump: np.ndarray,
    is_crisis: np.ndarray,
    water_flows: np.ndarray,
    quota: np.ndarray,
    DOE = 15,
    DCR = 10,
    max_subvention_weight = -0.5,#-3.0
    max_fine_weight = 0.1,
    policy_rate = 0.5,
    policy_skew = 2.0,
    i_alpha = 0.9,
    subsidy_fine_ratio = 2.0,

) -> np.ndarray:
    """
    Continuous incentive policy using generalized logistic function
    """
    # overuse or underuse penalty
    # -ve values are subsidies and +values are fines
    # actors using less than quota will be get subsidies
    # actors using above quota will get fined

    # get crisis level from last run
    crisis_level = is_crisis[-1]

    # use remaining water to check if in crisis after pumping
    # this factors into penalizing actors actions with knowledge of prediction
    
    estimated_remaining_water = water_flows[-1] - water_pump.sum()
    if estimated_remaining_water <= DCR:
        next_crisis = 2
    elif estimated_remaining_water <= ((DCR + DOE) / 2):
        next_crisis = 1
    elif estimated_remaining_water <= DOE:
        next_crisis = 0
    else:
        next_crisis = -1

    
    # new crisis level
    # weights to determine focus on current or past crisis
    crisis_level = i_alpha * next_crisis + (1 - i_alpha) * crisis_level

    # check for ongoing crisis
    if is_crisis[-1] >= 1 and is_crisis[-2] >= 1:
        crisis_level = is_crisis[-1] + is_crisis[-2]

    # round to give integer
    hard_crisis_level = round(crisis_level, 0)

        # compute overuse
    overuse = water_pump - quota

    # get overuse _rate 
    overuse_rate = overuse / (quota + np.exp(2-actors_priority)) # regularize weight with actor's priority

    # compute overuse coefficients
    # overuse coeff switches focus between fairness and ecological awareness
    c = hard_crisis_level + 1
    overuse_coef = np.exp(-policy_skew * c)

    # get raw overuse score
    p = overuse_coef * overuse_rate + (1 - overuse_coef) * overuse

    # convert score to max_subvention - max_fine range using generalized logistic function
    penalty_rate = max_subvention_weight + ((max_fine_weight - max_subvention_weight) / (1 + np.exp(-p * (c+1) * policy_rate)))

    # diverge rate of subsidy and penalty 
    penalty_rate = np.where(penalty_rate < 0, penalty_rate * subsidy_fine_ratio, penalty_rate)

    # compute subventions and fines
    incentives = avg_incomes  * penalty_rate  
   
    return incentives

