import numpy as np
import matplotlib.pyplot as plt

class DataGenerator:
    def __init__(self, n=50000, random_state=42):
        self.n = n
        self.random_state = random_state
        self.xl = []
        self.yl = []
        self.masks = []

        # Data generation
        np.random.seed(self.random_state)
        cl = np.random.rand(5)<.25  # left to be aligned with provided data
        
        for _ in range(self.n):
            cl = np.random.rand(5)<.25
            row, mask = self.createRow(np.random.randint(40,60), cl)
            self.xl.append(row)
            self.yl.append(cl)
            self.masks.append(mask)

    
    def createRow(self, n, classes):
        mask = [None for _ in range(len(classes))]
        base = np.sin(np.linspace((np.random.rand(3)),(np.random.rand(3) + np.array([10,15,7])), n))
        if classes[0] > 0:
            x = np.random.randint(0, n)
            base[x, 0] += 2
            mask[0] = [(0, (x, x))]
        if classes[1] > 0:
            x = np.random.randint(0, n)
            base[x, 1] -= 2
            mask[1] = [(1, (x, x))]
        if classes[2] > 0:
            x = np.random.randint(0, n-5)
            base[x:x+4,2] = 0
            mask[2] = [(2, (x, x+3))]
        if classes[3] > 0:
            x = np.random.randint(0, n-10)
            base[x:x+8,1] += 1.5
            mask[3] = [(1, (x, x+7))]
        if classes[4] > 0:
            x = np.random.randint(0, n-7)
            base[x:x+6,0] += 1.5
            base[x:x+6,2] -= 1.5
            mask[4] = [(0, (x, x+5)), (2, (x, x+5))]
        base += np.random.rand(*base.shape)*.2
        
        return base, mask
    

    def visualize_data(self, num_samples=3):
        for i in range(num_samples):
            plt.plot(self.xl[i])
            print(self.yl[i])
            plt.show()

    
    def get_data(self):
        return self.xl, self.yl, self.masks