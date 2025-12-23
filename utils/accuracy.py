import torch.utils.data
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

class RankAccuracy(Metric):

    def __init__(self, output_transform=lambda x: x, device='cpu'):
        self._num_correct = None
        self._num_examples = None
        super(RankAccuracy, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0
        super(RankAccuracy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        (rank_left, rank_right, label) = output
        rank_left = rank_left.squeeze()
        rank_right = rank_right.squeeze()

        # Compute accuracy for non ties
        index_mask = label != 0                                 # Select non-ties
        aux_label = -1 * label[index_mask]                      # Invert label (-1 for right winning, 1 for left)
        diff = rank_left[index_mask] - rank_right[index_mask]   # Compute rank difference
        '''
        # ---- DEBUG BEGIN ----
        if aux_label.numel() > 0:
            
            # simple sign-based correctness (what RankAccuracy uses)
            correct = ((aux_label == 1) & (diff > 0)) | ((aux_label == -1) & (diff < 0))
            batch_acc = correct.float().mean().item()
    
            frac_zero = (diff == 0).float().mean().item()
            mean_diff = diff.mean().item()
            std_diff = diff.std().item() if diff.numel() > 1 else 0.0
     
                print(
                f"[DEBUG RankAccuracy] "
                f"batch_non_ties={aux_label.numel()} "
                f"batch_acc={batch_acc:.4f} "
                f"mean_diff={mean_diff:.4f} std_diff={std_diff:.4f} "
                f"frac_diff_zero={frac_zero:.4f}"
            )
            
        
        else:
            print("[DEBUG RankAccuracy] batch has NO non-ties")
        # ---- DEBUG END ----
        '''
        correct_left = (aux_label == 1) & (diff > 0)            # Compute correct rank for left choices
        correct_right = (aux_label == -1) & (diff < 0)          # Compute correct rank for right choices
        self._num_correct += torch.sum(correct_left + correct_right).item()
        self._num_examples += aux_label.size()[0]



    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._num_correct / self._num_examples


class RankAccuracy_withMargin(Metric):
    def __init__(self, output_transform=lambda x: x, device='cpu'):
        self._num_correct = None
        self._num_examples = None
        super(RankAccuracy_withMargin, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0
        super(RankAccuracy_withMargin, self).reset()

    @reinit__is_reduced
    def update(self, output):
        (rank_left, rank_right, label, margin) = output
        rank_left = rank_left.squeeze()
        rank_right = rank_right.squeeze()

        # Compute accuracy for non ties
        index_mask = label != 0                                   # Select non-ties
        aux_label = -1 * label[index_mask]                        # Invert label (-1 for right winning, 1 for left)
        diff = rank_left[index_mask] - rank_right[index_mask]     # Compute rank difference
        correct_left = (aux_label == 1) & (diff > margin)         # Compute correct rank for left choices
        correct_right = (aux_label == -1) & (diff < -1 * margin)  # Compute correct rank for right choices

        self._num_correct += torch.sum(correct_left + correct_right).item()
        self._num_examples += aux_label.size()[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._num_correct / self._num_examples

class RankAccuracy_ties(Metric):

    def __init__(self, output_transform=lambda x: x, device='cpu'):
        self._num_correct = None
        self._num_examples = None
        super(RankAccuracy_ties, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0
        super(RankAccuracy_ties, self).reset()

    @reinit__is_reduced
    def update(self, output):
        (rank_left, rank_right, label, margin) = output
        rank_left = rank_left.squeeze()
        rank_right = rank_right.squeeze()

        index_mask = label == 0                                       # Select ties
        aux_label = label[index_mask]                                 # Get labels
        diff = rank_left[index_mask] - rank_right[index_mask]         # Compute rank difference
        correct_ties = (aux_label == 0) & (torch.abs(diff) < margin)  # Compute correct rank for left choices
        self._num_correct += torch.sum(correct_ties).item()
        self._num_examples += aux_label.size()[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._num_correct / self._num_examples


if __name__ == '__main__':
    import torch
    torch.manual_seed(8)

    m = RankAccuracy()
    output = (
        torch.Tensor([1, 2, 3, 1]),
        torch.Tensor([0, 3, 1, 3]),
        torch.Tensor([1, -1, 0, 1])
    )

    m.update(output)
    m.update(output)
    res = m.compute()

    print(m._num_correct, m._num_examples, res)