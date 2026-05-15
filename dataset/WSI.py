import os.path as osp

import torch
from torch.utils.data import Dataset


class WSIClf(Dataset):
    def __init__(self, patient_ids, feat_path, table_path, feat_format):
        super().__init__()
        self.read_path = feat_path
        self.read_format = feat_format

        from .data_utils import retrieve_from_table_clf
        self.sids, self.sid2pid, self.sid2label, self.pid2label = retrieve_from_table_clf(
            patient_ids,
            table_path,
            ret=["sid", "sid2pid", "sid2label", "pid2label"],
            level="slide",
        )

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, index):
        sid = self.sids[index]
        label = int(self.sid2label[sid])

        data_index = torch.tensor([index], dtype=torch.int)
        label = torch.tensor([label], dtype=torch.long)

        from .data_utils import read_patch_data
        full_path = osp.join(self.read_path, f"{sid}.{self.read_format}")
        feats = read_patch_data(full_path, dtype="torch").to(torch.float)
        return data_index, feats, label
