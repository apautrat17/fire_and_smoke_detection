def box_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1 - w1 / 2, x2 - w2 / 2)
    yi1 = max(y1 - h1 / 2, y2 - h2 / 2)

    xi2 = min(x1 + w1 / 2, x2 + w2 / 2)
    yi2 = min(y1 + h1 / 2, y2 + h2 / 2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    union = w1 * h1 + w2 * h2 - inter

    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_threshold=0.5):

    if len(boxes) == 0:
        return []

    indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    keep = []

    while indices:

        current = indices.pop(0)
        keep.append(current)

        filtered = []

        for idx in indices:

            iou = box_iou(boxes[current], boxes[idx])

            if iou < iou_threshold:
                filtered.append(idx)

        indices = filtered

    return keep


def compute_metrics(pred_boxes, gt_boxes, iou_thresh=0.5):

    """
    pred_boxes format:
    [b, cls, conf, x, y, w, h]
    """

    tp, fp = 0, 0
    matched = set()

    # sort by confidence
    pred_boxes = sorted(pred_boxes, key=lambda x: x[2], reverse=True)

    for pred in pred_boxes:

        pb, pcls, conf, px, py, pw, ph = pred

        best_iou = 0
        best_gt_idx = None

        for i, gt in enumerate(gt_boxes):

            gb, gcls, gx, gy, gw, gh = gt

            if pb != gb or pcls != gcls:
                continue

            iou = box_iou(
                [px, py, pw, ph],
                [gx, gy, gw, gh]
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_thresh and best_gt_idx not in matched:
            tp += 1
            matched.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-6)
    )

    # simplified AP approximation
    ap = precision * recall

    return precision, recall, f1, ap