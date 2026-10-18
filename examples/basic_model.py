import numpy as np
import torch
from symmclimatenet import build_model

coords = np.array([[37.98,23.72],[40.64,22.94],[39.64,22.42],[38.25,21.73],
                   [39.67,20.85],[40.85,25.87],[40.47,25.52],[40.90,24.43]], dtype=np.float32)
model = build_model(n_features=7, station_coordinates=coords, target_feature_index=0)
x = torch.zeros(2, 30, 8, 7)
out = model(x, target_station_index=0)
print(out["quantiles"].shape)
print(float(out["operator_symmetry_error"]))
