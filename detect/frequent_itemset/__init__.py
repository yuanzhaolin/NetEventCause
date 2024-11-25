
def fill_missing_cols(df):
    # missing_rows = df['cause-1'].isnull()
    # df.loc[df['cause-1'].isnull(), ] = df.loc[df['cause-1'].isnull()]
    df = df.copy()
    missing_rows = df['cause-1'].isnull()
    df.loc[missing_rows, 'cause-1'] = df.loc[missing_rows, '_id']
    df.loc[missing_rows, 'cause-1-score'] = 100

    df['root_prob'] = (100-df['cause-1-score'])/100
    df['modified_root_prob'] = df['root_prob']
    df['root_alarm'] = (df['root_prob'] > 0.5).tolist()
    df['identified'] = True
    df['algorithm'] = 'frequent items'
    df['root-1'] = df['cause-1']
    # df['algorithm'] = 'frequent items'
    return df
