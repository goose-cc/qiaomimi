from types import SimpleNamespace
import torch
from TransformerInverse import ParametricInverseTransformer1D
from mc_online_physics import OnlinePhysics
from mc_parametric import normalized_parameter_mse


def main():
    torch.manual_seed(7)
    args=SimpleNamespace(
        physics_dtype='float32', noise_level=0.0, data_scale=160000.0,
        shift=400.0, s_min=0.1764, s_max=6.0, q2_min=-100.0, q2_max=-6.0,
        output_points=1000, input_points=100, integration_points=64,
    )
    physics=OnlinePhysics(args,torch.device('cpu'))
    true=torch.tensor([[0.15,0.02,-0.01,0.9,0.2],[0.10,0.01,0.0,1.5,0.4]],dtype=torch.float32)
    _,g=physics.make_clean_batch(true)
    model=ParametricInverseTransformer1D(
        input_length=100, output_length=1000, d_model=32, nhead=4,
        num_encoder_layers=2, dim_feedforward=64, dropout=0.0,
    )
    f,p=model.forward_with_parameters(g.unsqueeze(1))
    loss=normalized_parameter_mse(
        p,true,model.parameter_lower,model.parameter_upper,
        torch.ones(5),
    )
    loss.backward()
    grad=sum(float(x.grad.abs().sum()) for x in model.parameters() if x.grad is not None)
    assert f.shape==(2,1,1000)
    assert p.shape==(2,5)
    assert torch.isfinite(loss) and grad>0
    print('EXP7 PARAMETRIC TEST PASSED')
    print('f shape:',tuple(f.shape))
    print('parameter shape:',tuple(p.shape))
    print('parameters:',sum(x.numel() for x in model.parameters()))


if __name__=='__main__': main()
