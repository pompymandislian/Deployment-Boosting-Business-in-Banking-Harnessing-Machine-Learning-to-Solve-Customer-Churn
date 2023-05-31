import pylab
import seaborn as sns
import matplotlib.pyplot as plt

class Visualization_outlier:
    """
    This class can make help for visualization data outlier
    """
    def __init__(self, data):
        self.data = data
        self.numeric_cols = self.data.select_dtypes(include=['number']).columns

    def kdeplot(self):
        """
        This function using for check distribution data
        """
        fig, axs = plt.subplots(nrows=len(self.numeric_cols), figsize=(8, len(self.numeric_cols)*4))
        for i, col in enumerate(self.numeric_cols):
            kdeplot = sns.kdeplot(data=self.data, x=col, color="blue", ax=axs[i])
        return kdeplot
    
    def boxplot(self):
        """
        this function using for check data outlier 
        """
        fig, axs = plt.subplots(nrows=len(self.numeric_cols), figsize=(8, len(self.numeric_cols)*4))
        for i, col in enumerate(self.numeric_cols):
            boxplot = sns.boxplot(data=self.data, x=col, color="blue", ax=axs[i])
        return boxplot   