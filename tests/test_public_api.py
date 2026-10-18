import numpy as np
import torch
import symmclimatenet

def test_version_and_lock():
    assert symmclimatenet.__version__ == "1.0.0"
    assert symmclimatenet.locked_spec_dict()["variant"] == "set_temporal_plus_symmetric_spatial"

def test_public_model_is_symmetric_and_permutation_invariant():
    torch.manual_seed(13)
    coords=np.array([[37.98,23.72],[40.64,22.94],[39.64,22.42],[38.25,21.73]],dtype=np.float32)
    m=symmclimatenet.build_model(n_features=3,station_coordinates=coords,target_feature_index=0).eval()
    x=torch.randn(2,30,4,3)
    with torch.no_grad():
        y=m(x,target_station_index=1)
    assert float(y["operator_symmetry_error"]) <= 1e-6
    p=torch.tensor([2,0,3,1])
    x2=x[:,:,p,:]
    c2=coords[p.numpy()]
    new_target=int((p==1).nonzero(as_tuple=False).item())
    with torch.no_grad():
        y2=m(x2,target_station_index=new_target,coordinates=c2)
    assert torch.allclose(y["quantiles"],y2["quantiles"],atol=2e-5,rtol=2e-5)
