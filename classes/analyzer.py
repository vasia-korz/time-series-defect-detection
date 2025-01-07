import numpy as np

class Analyzer:
    def __init__(self, explainer, x_test, y_true, masks):
        self.explainer = explainer
        self.x_test = x_test
        self.y_true = y_true
        self.masks = masks


    def iou_overall(self):
        union, intersection = 0, 0

        for i, explanation in enumerate(self.explainer.explanations):
            for j in range(len(explanation["feature"])):
                for mask in (self.masks[i][j] if self.masks[i][j] is not None else [None]):
                    if explanation["pred"][0][j] == 1:  # model predicts the defect of class j
                        left, right = explanation["left"][j], explanation["right"][j]
                        
                        if mask is None:  # in reality there is no defect of class j
                            union += (right - left + 1)
                        else:
                            mask_left, mask_right = mask[1]

                            if self._is_intersecting(left, right, mask_left, mask_right):  # there is an intersection
                                intersection += self._intersection(left, right, mask_left, mask_right)
                                union += self._union(left, right, mask_left, mask_right)
                            else:
                                union += (right - left + 1)
                                union += (mask_right - mask_left + 1)
                    else:
                        if mask is not None:  # defect of class j was not indentified by the classifier
                            mask_left, mask_right = mask[1]
                            union += (mask_right - mask_left + 1)

        if union == 0:
            return 0
        
        return intersection / union
    

    def iou(self):
        ious = []

        for i, explanation in enumerate(self.explainer.explanations):
            for j in range(len(explanation["feature"])):
                iou_curr = []

                for mask in (self.masks[i][j] if self.masks[i][j] is not None else [None]):
                    if explanation["feature"][j] == -1 and mask is None:
                        continue

                    iou_curr.append(self._iou_single(explanation["feature"][j], explanation["left"][j], explanation["right"][j], mask))
                
                if len(iou_curr):
                    iou = sum(iou_curr) / len(iou_curr)
                    ious.append(iou)

        return sum(ious) / len(ious) if len(ious) > 0 else 0
    

    def class_based_iou(self):
        ious = [[] for _ in range(len(self.explainer.explanations[0]["feature"]))]

        for i, explanation in enumerate(self.explainer.explanations):
            for j in range(len(explanation["feature"])):
                iou_curr = []

                for mask in (self.masks[i][j] if self.masks[i][j] is not None else [None]):
                    if explanation["feature"][j] == -1 and mask is None:
                        continue

                    iou_curr.append(self._iou_single(explanation["feature"][j], explanation["left"][j], explanation["right"][j], mask))
                
                if len(iou_curr):
                    iou = sum(iou_curr) / len(iou_curr)
                    ious[j].append(iou)

        return [(sum(iou) / len(iou) if len(iou) > 0 else 0) for iou in ious]
    
    
    def _iou_single(self, feature, left, right, mask):
        if mask is None:
            return 0
        
        mask_feature, (mask_left, mask_right) = mask

        if feature != mask_feature:
            return 0
        
        intersection = self._intersection(left, right, mask_left, mask_right)
        union = self._union(left, right, mask_left, mask_right)
        
        if union == 0:
            return 0
        
        return intersection / union
            

    def _union(self, left, right, mask_left, mask_right):
        if not self._is_intersecting(left, right, mask_left, mask_right):
            return right + mask_right - left - mask_left + 2
        
        return max(right, mask_right) - min(left, mask_left) + 1 
    

    def _intersection(self, left, right, mask_left, mask_right):
        if not self._is_intersecting(left, right, mask_left, mask_right):
            return 0
        
        return min(right, mask_right) - max(left, mask_left) + 1
    

    def _is_intersecting(self, left, right, mask_left, mask_right):
        return not (right < mask_left or left > mask_right)
         

    def accuracy(self, debug=False):
        correct = 0
        overall = 0
        
        for i, explanation in enumerate(self.explainer.explanations):
            for j in range(len(explanation["feature"])):
                for mask in (self.masks[i][j] if self.masks[i][j] is not None else [None]):
                    if explanation["pred"][0][j] == 1:  # model predicts the defect of class j
                        feature, left, right = explanation["feature"][j], explanation["left"][j], explanation["right"][j]
                        
                        if mask is None or mask[0] != feature:  # in reality there is no defect of class j or different feature predicted
                            overall += 1
                        else:
                            mask_left, mask_right = mask[1]

                            if self._is_intersecting(left, right, mask_left, mask_right):  # there is an intersection
                                correct += 1
                                overall += 1
                            else:
                                overall += 1
                    else:
                        if mask is not None:  # defect of class j was not indentified by the classifier
                            overall += 1
        
        if debug:
            print(f"Correct: {correct}, overall: {overall}")

        return correct / overall if overall > 0 else 0
    

    def class_based_accuracy(self, debug=False):
        correct = [0 for _ in range(len(self.explainer.explanations[0]["feature"]))]
        overall = [0 for _ in range(len(self.explainer.explanations[0]["feature"]))]
        
        for i, explanation in enumerate(self.explainer.explanations):
            for j in range(len(explanation["feature"])):
                for mask in (self.masks[i][j] if self.masks[i][j] is not None else [None]):
                    if explanation["pred"][0][j] == 1:  # model predicts the defect of class j
                        feature, left, right = explanation["feature"][j], explanation["left"][j], explanation["right"][j]
                        
                        if mask is None or mask[0] != feature:  # in reality there is no defect of class j or different feature predicted
                            overall[j] += 1
                        else:
                            mask_left, mask_right = mask[1]

                            if self._is_intersecting(left, right, mask_left, mask_right):  # there is an intersection
                                correct[j] += 1
                                overall[j] += 1
                            else:
                                overall[j] += 1
                    else:
                        if mask is not None:  # defect of class j was not indentified by the classifier
                            overall[j] += 1
        
        if debug:
            print(f"Correct: {correct}, overall: {overall}")

        return [(c / o if o > 0 else 0) for c, o in zip(correct, overall)]
    

    def analyze(self, scale=3):
        n_classes = self.explainer.model.n_classes

        solutions = {
            "pos": {i: [] for i in range(n_classes)},  # correct predictions
            "neg": {i: [] for i in range(n_classes)}   # incorrect predictions
        }

        for id, explanation in enumerate(self.explainer.explanations):
            for i in range(n_classes):
                masks = self.masks[id][i]

                pred = explanation["pred"][0][i]
                feature, left, right = explanation["feature"][i], explanation["left"][i], explanation["right"][i]

                if pred == 1:
                    for mask in masks if masks is not None else [None]:
                        if mask is not None and mask[0] == feature:
                            mask_left, mask_right = mask[1]

                            if self._is_intersecting(left, right, mask_left, mask_right):
                                solutions["pos"][i].append((id, feature, left, right))
                            else:
                                solutions["neg"][i].append((id, feature, left, right))

        for i in range(n_classes):
            pos_examples = solutions["pos"][i][:scale]
            if pos_examples:
                x_list, feature_indices, left_list, right_list = zip(*[
                    (self.x_test[id][:, feature], feature, left, right) for id, feature, left, right in pos_examples
                ])
                print(f"Class {i + 1}: Correct Predictions")
                x_list = np.array(x_list)
                x_list = [sub[sub != self.explainer.model.padding_value] for sub in x_list]
                self.explainer._plot_features_with_highlights(x_list, feature_indices, left_list, right_list)

            neg_examples = solutions["neg"][i][:scale]
            if neg_examples:
                print(f"Class {i + 1}: Incorrect Predictions")
                x_list, feature_indices, left_list, right_list = zip(*[
                    (self.x_test[id][:, feature], feature, left, right) for id, feature, left, right in neg_examples if feature != -1
                ])
                x_list = np.array(x_list)
                x_list = [sub[sub != self.explainer.model.padding_value] for sub in x_list]
                self.explainer._plot_features_with_highlights(x_list, feature_indices, left_list, right_list)
    