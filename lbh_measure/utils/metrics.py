import numpy as np 

def compute_iou(self, pred_np, seg_np):
        IoUs = []
        for pred, seg in zip(pred_np, seg_np):
            pred = pred[0]
            seg = seg[0]
            I_all = np.zeros(2)
            U_all = np.zeros(2)
            for sem_idx in range(seg.shape[0]):
                for sem in range(2):
                    I = np.sum(np.logical_and(pred[sem_idx] == sem, seg[sem_idx] == sem))
                    U = np.sum(np.logical_or(pred[sem_idx] == sem, seg[sem_idx] == sem))
                    I_all[sem] += I
                    U_all[sem] += U
            
            IoUs.append(I_all / U_all )
        IoUs = np.array(IoUs)
        return IoUs.mean()