import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

class Explainer:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        self.y_test = y_test
        self.explanations = []

    
    def gen_single(self, id=None, integrated_grads=True, debug=True):
        pred, grads, features, poses, lefts, rights = None, None, [], [], [], []
        flag = False
        if id is None:
            id = np.random.randint(len(self.x_test))
            flag = True

        x = self.x_test[id:id+1]
        pred = self.model._apply_threshold(self.model.model.predict(x, verbose=0)).astype(np.int32)
        if integrated_grads:
            grads = self._get_integrated_grads(x, pred)
        else:
            grads = self._get_grads(x, pred)

        if debug:
            print("Predicted classes:", pred[0])
            print("Expected classes:", self.y_test[id].astype(np.int32), "\n")

        if (pred > 0).sum() == 0 and flag:
            print("There are no defects. Running one more test.")
            self.gen_single()
            return

        x, grads, _ = self._remove_padding(x, grads)

        for i in range(len(pred[0])):
            if pred[0][i]:
                feature, pos, left, right = self._find_region_with_spike(grads[i])
                features.append(feature)
                poses.append(pos)
                lefts.append(left)
                rights.append(right)
                if debug:
                    print(f"Deffect class {i}")
                    self._visualize_separate(x)
                    self._visualize_separate(grads[i], title="Gradient / Importance of Feature")
                    self._plot_features_with_highlights([x[:, feature]], [feature], [left], [right])
            else:
                features.append(-1)
                poses.append(-1)
                lefts.append(-1)
                rights.append(-1)
        
        return pred, grads, features, poses, lefts, rights

    
    def gen_explanations(self, n=None, integrated_grads=False):
        self.explanations = []

        if n is None:  # generate all explanations
            n = len(self.x_test)

        for id in tqdm(range(0, n), desc="Generating Explanations"):
            pred, grads, feature, pos, left, right = self.gen_single(id, integrated_grads=integrated_grads, debug=False)
            self.explanations.append({
                "pred": pred,
                "grads": grads,
                "feature": feature,
                "pos": pos,
                "left": left,
                "right": right
            })

    
    def _get_grads(self, x, pred):
        grads = []
        
        for i in range(len(pred[0])):  # for each class decision
            if pred[0][i] == 1:
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    preds = self.model.model(x)
                    class_pred = preds[:, i]
                grad = tape.gradient(class_pred, x)
            else:
                grad = None

            if grad is not None:
                grads.append(grad.numpy())
            else:
                grads.append(np.zeros_like(x))

        return np.stack(grads, axis=0)
    

    def _get_integrated_grads(self, x, pred):
        grads = []

        for i in range(len(pred[0])):  # for each class decision
            if pred[0][i] == 1:
                grad = self._integrated_gradients(x, i)
            else:
                grad = None

            if grad is not None:
                grads.append(grad.numpy())
            else:
                grads.append(np.zeros_like(x))

        return np.stack(grads, axis=0)
    

    def _integrated_gradients(self, x, target_class_idx, baseline=None, steps=50):
        if baseline is None:
            baseline = tf.zeros_like(x)

        alphas = tf.linspace(0.0, 1.0, steps+1)
        interpolated_inputs = [
            baseline + alpha * (x - baseline) for alpha in alphas
        ]
        interpolated_inputs = tf.concat(interpolated_inputs, axis=0)
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated_inputs)
            preds = tf.cast(self.model.model.predict(interpolated_inputs, verbose=0), dtype=tf.float32)
            target_scores = preds[:, target_class_idx]
        
        grads = tape.gradient(target_scores, interpolated_inputs)
        
        grads = tf.reshape(grads, (steps+1, x.shape[0], x.shape[1], x.shape[2]))
        
        avg_grads = tf.reduce_mean(grads[:-1] + grads[1:], axis=0) / 2.0
        integrated_grads = (x - baseline) * avg_grads
        
        return integrated_grads

    
    def _visualize_separate(self, x, title='Feature'):
        global_min = np.min(x)
        global_max = np.max(x)

        plt.figure(figsize=(15, 4))
        _, axs = plt.subplots(1, 3, figsize=(15, 4))
        
        axs[0].plot(x[..., 0])
        axs[0].set_title(f"{title} 1")
        axs[0].set_ylim(global_min, global_max)
        
        axs[1].plot(x[..., 1])
        axs[1].set_title(f"{title} 2")
        axs[1].set_ylim(global_min, global_max)
        
        axs[2].plot(x[..., 2])
        axs[2].set_title(f"{title} 3")
        axs[2].set_ylim(global_min, global_max)
        
        plt.tight_layout()
        plt.show()


    def _remove_padding_x(self, x):
        m = np.array((x != [self.model.padding_value for _ in range(3)])).all(axis=2).sum()
       
        if self.model.padding == 'post':
            return np.array(x[:, :m]).reshape(m, 3)
        
        return np.array(x[:, -m:]).reshape(m, 3)


    def _remove_padding(self, x, grads, border_cut=0):
        m = np.array((x != [self.model.padding_value for _ in range(3)])).all(axis=2).sum()
        m -= border_cut

        if self.model.padding == 'post':
            return np.array(x[:, :m]).reshape(m, 3), np.abs(grads.reshape(grads.shape[0], grads.shape[2], grads.shape[3])[:, :m, :]), m
        
        return np.array(x[:, -m:]).reshape(m, 3), np.abs(grads.reshape(grads.shape[0], grads.shape[2], grads.shape[3])[:, -m:, :]), m
    

    def _find_region_with_spike(self, grads_n, known_feature=None, threshold_scale=1.0, max_dip=0, expand_margin=0):
        if known_feature is None:
            idx = grads_n.flatten().argmax()
            feature = idx % self.model.n_features
            pos = idx // self.model.n_features
        else:
            idx = grads_n[:, known_feature].flatten().argmax()
            feature = known_feature
            pos = idx
        
        saliency_feature = grads_n[:, feature]

        threshold = saliency_feature.mean() * threshold_scale
        dip_threshold = threshold * (1 - max_dip)

        left, right = pos, pos
        while left > 0:
            if saliency_feature[left - 1] > threshold:
                left -= 1
            elif saliency_feature[left - 1] > dip_threshold and saliency_feature[left - 2] > saliency_feature[left - 1]:
                left -= 1
            else:
                break
        
        while right < len(saliency_feature) - 2:
            if saliency_feature[right + 1] > threshold:
                right += 1
            elif saliency_feature[right + 1] > dip_threshold and saliency_feature[right + 2] > saliency_feature[right + 1]:
                right += 1
            else:
                break

        left = max(0, left - expand_margin)
        right = min(len(saliency_feature) - 1, right + expand_margin)

        return feature, pos, left, right


    def _plot_features_with_highlights(self, x_list, feature_indices, left_list, right_list):
        n = len(x_list)
        if not (len(feature_indices) == len(left_list) == len(right_list) == n):
            raise ValueError("All input lists must have the same length.")

        fig, axs = plt.subplots(1, n, figsize=(5 * n, 4), constrained_layout=True)

        if n == 1:
            axs = [axs]

        for i, (x, feature_index, left, right) in enumerate(zip(x_list, feature_indices, left_list, right_list)):
            axs[i].plot(x, label=f"Feature {feature_index + 1}", color='blue')

            if left == right:
                axs[i].scatter(
                    left,
                    x[left],
                    color='red',
                    label="Defect Region",
                    zorder=5
                )
            else:
                axs[i].plot(
                    range(left, right + 1),
                    x[left:right + 1],
                    label="Defect Region",
                    color='red',
                    linewidth=2
                )

            axs[i].set_title(f"Feature {feature_index + 1} with Highlighted Region")
            axs[i].set_xlabel("Timestep")
            axs[i].set_ylabel("Value")
            axs[i].legend()
            axs[i].grid()

        plt.show()

