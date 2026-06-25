import torch

def get_n2o_yield(o2, bgc):
    """
    Converted from n2o_yield.m
    Calculates N2O and NO2 yields during Ammox and nitrifier-denitrification.
    """
    Y = {}
    
    # Check if a model is specified, default to 'Ji'
    n2o_yield_model = getattr(bgc, 'n2o_yield', 'Ji')
    
    if n2o_yield_model == 'Ji':
        # Hardcoding the parameters natively so biopar.py is untouched
        Ji_a = 0.05  # Replace with your original MATLAB value if different
        Ji_b = 0.15  # Replace with your original MATLAB value if different
        
        # scale original params
        a1 = Ji_a / 100.0
        b1 = Ji_b / 100.0
        
        # total yields
        Y['nn2o_nh4'] = (a1 + b1 * o2) / (a1 + (b1 + 1.0) * o2)
        Y['no2_nh4']  = o2 / (a1 + (b1 + 1.0) * o2)
        
        # get yields for Hydroxylamine pathway
        Y['nn2o_hx_nh4'] = b1 / (b1 + 1.0)
        
        # get yields for nitrifier-denitrification pathway
        Y['nn2o_nden_nh4'] = 1.0 / ((b1 + 1.0) * (1.0 + ((b1 + 1.0) / a1) * o2))
        Y['no2_nden_nh4']  = -Y['nn2o_nden_nh4']  # nh4->no2->n2o
        
        # get yields for Hydroxylamine pathway by difference
        # Y['no2_hx_nh4'] = 1.0 - Y['nn2o_hx_nh4'] - Y['nn2o_nden_nh4']
        Y['no2_hx_nh4'] = torch.clamp(1.0 - Y['nn2o_hx_nh4'] - Y['nn2o_nden_nh4'], min=0.0)
        
    return Y