def WoE(dataframe, ls_categorical, target, inplace = False):
    """
    Returns Transformed Dataframe, Weight of Evidence and Information Value in that order.
    
    • Description:
    
    The function returns a tuple of the the original DataFrame with it's WoEs.
    
    • Parameters:
    
    dataframe := A pandas Dataframe
    ls_categorical:= List of categorical columns
    target := Array of binary targets. The functions undertands 1 as good and 0 as bad.
    inplace := Indicates if the WoEs replace the original categorical values
    
    """
    #### -1.- INNIT
    import pandas as pd
    import numpy as np
    df = dataframe
    
    #### 0.- ERROR HANDLER
    NullValues = sum([0 if sum(pd.isnull(df[col])) == 0 else 1 for col in df[ls_categorical]])
    if NullValues > 0:
        raise ValueError('Columns must NOT contain NULL VALUES')
    
    #### 1.- 
    
    WoEs = {}
    IVs = {}
    for col in ls_categorical:

        ls = [(elem,
               df[(df[col] == elem) & (df[target] == 1)].shape[0]/df[df[target] == 1].shape[0],
               df[(df[col] == elem) & (df[target] == 0)].shape[0]/df[df[target] == 0].shape[0])\
              for elem in np.unique(df[col])]
            
        woename = 'WoE_'+col
        IV = 'IV'
        
        frame = pd.DataFrame(ls)
        frame.rename(columns = {0:col}, inplace = True)
        frame[woename] = np.log(frame[1]/frame[2])
        frame[woename] = np.where(np.isinf(frame[woename]),0,frame[woename])
        frame[IV] = (frame[1]-frame[2])*frame[woename]
        
        dataframe = dataframe.merge(frame[[col,woename]], how = 'left', on = [col])
        
        WoEs[col] = (frame)
        IVs[col] = sum(frame[IV])

        if inplace == True:
            dataframe = dataframe.drop(col, axis = 1)
            
    return(dataframe,WoEs,IVs)
