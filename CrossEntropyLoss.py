import torch


def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)


y_true = [[0, 0, 1],
          [0, 1, 0],
          [1, 0, 0]]
y_true = torch.tensor(y_true)

y_pred_1 = [[0.3, 0.3, 0.4],
            [0.3, 0.4, 0.3],
            [0.1, 0.2, 0.7]]
y_pred_1 = torch.tensor(y_pred_1)

y_pred_2 = [[0.1, 0.2, 0.7],
            [0.1, 0.7, 0.2],
            [0.3, 0.4, 0.3]]
y_pred_2 = torch.tensor(y_pred_2)

print(CrossEntropyLoss(y_true, y_pred_1))

print(CrossEntropyLoss(y_true, y_pred_2))

