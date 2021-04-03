def qlib_to_alphalens(qlib_factor_data):
    
    factor_data = qlib_factor_data.reset_index().rename(columns={'datetime':'date','instrument':'asset'})
    factor_data = factor_data.sort_values(by=['date','asset']).set_index(['date','asset'])
    factor_data = factor_data.dropna()
    factor_data['group'] = factor_data['group'].apply(int).apply(str)
    return factor_data