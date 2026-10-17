from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tabseq_torch.dl.padded_batch import PaddedTabBatch


def mcc_loading_test(
    dl: DataLoader,
    mcc: pd.DataFrame,
) -> str:
    ptb: PaddedTabBatch
    for ptb in iter(dl):
        for sidx, mcc_seq in enumerate(ptb["mcc"]):
            gid = ptb._groups[sidx]
            sl = ptb.lengths[sidx]
            if mcc_seq.sum() != mcc_seq[:sl].sum():
                return f"boken len group={gid}: {sl}"
            ptb.lengths[sidx]
            seq = mcc.loc[gid]["mcc"]
            expected_seq = mcc_seq[:sl].numpy()
            if not np.array_equal(seq, expected_seq):
                return f"boken group gid:{gid}\n{seq}\n{expected_seq}"
    return "Ok!"
