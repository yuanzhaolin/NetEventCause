import pandas as pd


class RootCauseFind:
    def __init__(self, df, root_col):
        df = df.copy()
        df.loc[df[root_col].isnull(), root_col] = df.loc[df[root_col].isnull()].index
        self.map = pd.DataFrame({'root': df[root_col].tolist()}, index=df.index)
        for _id, _ in self.map.iterrows():
            # pdb.set_trace()
            self.update(_id)
        self.root_col = root_col
        self.map = pd.concat([self.map, pd.DataFrame([{'root': None}], index=['unknown'])])

    def update(self, _id):
        root = self.map.loc[_id]['root']
        if root == _id:
            return

        self.update(root)
        self.map.loc[_id, 'root'] = self.map.loc[root, 'root']

    def find_root(self, _ids):
        if isinstance(_ids, list):
            _ids = ['unknown' if x is None else x for x in _ids]
        return self.map.loc[_ids, 'root']
