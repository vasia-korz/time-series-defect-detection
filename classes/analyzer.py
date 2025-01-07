class Analyzer:
    def __init__(self, explanations, y_true, masks):
        self.explanations = explanations
        self.y_true = y_true
        self.masks = masks


    def iou_overall(self):
        union, intersection = 0, 0

        for i, explanation in enumerate(self.explanations):
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

        for i, explanation in enumerate(self.explanations):
            for j in range(len(explanation["feature"])):
                for mask in (self.masks[i][j] if self.masks[i][j] is not None else [None]):
                    if explanation["feature"][j] == -1 and mask is None:
                        continue

                    iou = self._iou_single(explanation["feature"][j], explanation["left"][j], explanation["right"][j], mask)
                    ious.append(iou)
        
        return sum(ious) / len(ious)
    
    
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
        
        for i, explanation in enumerate(self.explanations):
            for j in range(len(explanation["feature"])):
                for mask in (self.masks[i][j] if self.masks[i][j] is not None else [None]):
                    if explanation["pred"][0][j] == 1:  # model predicts the defect of class j
                        left, right = explanation["left"][j], explanation["right"][j]
                        
                        if mask is None:  # in reality there is no defect of class j
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

        return correct / overall
    

    def class_based_accuracy(self, debug=False):
        correct = [0 for _ in range(len(self.explanations[0]["feature"]))]
        overall = [0 for _ in range(len(self.explanations[0]["feature"]))]
        
        for i, explanation in enumerate(self.explanations):
            for j in range(len(explanation["feature"])):
                for mask in (self.masks[i][j] if self.masks[i][j] is not None else [None]):
                    if explanation["pred"][0][j] == 1:  # model predicts the defect of class j
                        left, right = explanation["left"][j], explanation["right"][j]
                        
                        if mask is None:  # in reality there is no defect of class j
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

        return [c / o for c, o in zip(correct, overall)]
    