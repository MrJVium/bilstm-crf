import torch
import torch.nn as nn

def log_sum_exp(x):
    # max 会缩减一维
    max_score = x.max(dim=-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()

IMPOSSIBLE = -1e4


class CRF(nn.Module):

    def __init__(self, in_features, num_tags):
        super(CRF, self).__init__()

        self.num_tags = num_tags + 2
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(in_features, self.num_tags)

        # transition factor, Tij mean transition from j to i
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

    def forward(self, features, masks):
        """
        Args:
            self:
            features: [b, len, c], batch of unary scores
            masks: [b, l] masks

        Returns: (best_score, best_paths)
            best_score: [b]
            best_paths: [b, len]
        """
        features = self.fc(features)
        return self.__viterbi_decode(features, masks[:, :features.size(1)].float())

    def loss(self, features, ys, masks):
        """

        Args:
            features: [b, len, d] 还是 hidden * 2
            ys: tags, [b, len]
            masks: [b, len]

        Returns: loss

        """
        features = self.fc(features)
        L = features.size(1)
        masks_ = masks[:, :L].float()

        forward_score = self.__forward_algorithm(features, masks_)
        gold_score = self.__score_sentence(features, ys[:, :L].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss

    def __score_sentence(self, features, tags, masks):
        B, L, C = features.shape

        emit_score = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)
        start_tags = torch.full((B, 1), self.start_idx, dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tags, tags], dim=1)
        trans_score = self.transitions[tags[:, 1:], tags[:, :-1]]

        last_tags = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)
        last_score = self.transitions[self.stop_idx, last_tags]

        score = ((emit_score + trans_score) * masks).sum(1) + last_score
        return score

    def __viterbi_decode(self, features, masks):
        B, L, C = features.shape

        bps = torch.full((B, L, C), dtype=torch.long, device=features.device)
        max_scores = torch.full((B, C), IMPOSSIBLE, dtype=torch.long, device=features.device)
        max_scores[:, self.start_idx] = 0

        for t in range(L):
            masks_t = masks[:, t].unsqueeze(1)
            emit_scores = features[:, t]   # 不用放缩  找出最大的直接加上就行

            acc_score_t = max_scores.unsqueeze(1) + self.transitions
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_scores
            max_scores = acc_score_t * masks_t + max_scores * (1 - masks_t)
        max_scores += self.transitions[self.stop_idx]
        best_score, best_tag = max_scores.max(dim=-1)

        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            best_path = [best_tag_b]
            seq_len = int(masks[b].sum().item())

            # 长度是 seq_len, 最后一个tag的下标是 seq_len-1, bps_t[b, seq_len-1] 就是倒数第二个，所以从这开始
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def __forward_algorithm(self, features, masks):

        B, L, C = features.shape
        scores = torch.full((B, C), IMPOSSIBLE, device=features.device)
        scores[:, self.start_idx] = 0
        trans_score = self.transitions.unsqueeze(0)

        for t in range(L):
            emit_scores = features[:, t].unsqueeze(-1)
            scores_t = scores.unsqueeze(1) + trans_score + emit_scores
            scores_t = log_sum_exp(scores_t)

            masks_t = masks[:, t].unsqueeze(1)
            scores = scores_t * masks_t + scores * (1 - masks_t)
        scores = log_sum_exp(scores + self.transitions[self.stop_idx])
        return scores