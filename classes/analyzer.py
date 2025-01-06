class Analyzer:
    def __init__(self, explanations, y_true, masks):
        self.explanations = explanations
        self.y_true = y_true
        self.masks = masks


    def iou(self):
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
                                intersection += min(right, mask_right) - max(left, mask_left) + 1
                                union += max(right, mask_right) - min(left, mask_left) + 1
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