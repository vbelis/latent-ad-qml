# Module that defines the main object KernelMachine, used to implement Quantum
# Support Vector Machines (QSVM) and classical Support Vector Machines (SVM) in
# this work.
from sklearn.svm import SVC

class KernelMachine(SVC):
    """Kernel machine class. Can be either a quantum kernel-based model (QSVM),
    or a classical SVM.

    Attributes:
        hp (dict): Hyperaparameters that define the kernel model.
    """
    def __init__(self, hpars: dict):
        """
        Args:
            hpars: Hyperparameters of the model and configuration parameters
                   for the training.
        """
        self.C = hpars["c_param"]
        self.gamma = hpars["gamma"]
        self.is_quantum = hpars["quantum"]
        
        self.hp = {
            "C": self.C,
            "gamma": self.gamma,
        }
        if not self.is_quantum:
            super().__init__(kernel='rbf', gamma=self.gamma, C=self.C)
        else:
            super().__init__(kernel='precomputed', C=self.C)
            self.feature_map 
            self.kernel = self.config_quantum_kernel()
        
    def config_quantum_kernel(self, args):

        if self.is_quantum: 
            return QuantumKernel(self.feature_map, self.quantum_instance,)
    
    # qsvm.fit(train_features, train_labels)