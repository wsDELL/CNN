from MiniFramework import *

hp = HyperParameters(0.1, optimizer_name=OptimizerName.Adam)
con=ConLayer(2, 4, 3, hp=hp)
data = np.random.randn([2, 2, 9, 9])
ConLayer.
