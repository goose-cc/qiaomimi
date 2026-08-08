from types import SimpleNamespace
import torch
from TransformerInverse import PeakParametricInverseTransformer1D
from mc_online_physics import OnlinePhysics
from mc_parametric_strong import strong_parameter_mse,relative_mse,gradient_relative_mse,log_width_mse,log_peak_height_mse

def main():
    torch.manual_seed(8); device=torch.device('cpu')
    a=SimpleNamespace(physics_dtype='float32',noise_level=0.0,data_scale=160000.0,shift=400.0,s_min=0.1764,s_max=6.0,q2_min=-100.0,q2_max=-6.0,output_points=1000,input_points=100,integration_points=64)
    physics=OnlinePhysics(a,device)
    true=torch.tensor([[0.15,0.02,-0.01,0.9,0.02],[0.10,0.01,0.0,1.5,0.2]],dtype=torch.float32)
    tt,tr,_=physics.components(true); g=physics.forward_from_parameters(true)
    model=PeakParametricInverseTransformer1D(input_length=100,output_length=1000,d_model=32,nhead=4,num_encoder_layers=2,dim_feedforward=64,dropout=0.0,gamma_log_floor=1e-5)
    pred=model.predict_parameters(g.unsqueeze(1)); pt,pr,_=physics.components(pred); pg=physics.forward_from_parameters(pred)
    w=torch.tensor([3.,1.,1.,5.,6.]); loss=strong_parameter_mse(pred,true,model.parameter_lower,model.parameter_upper,w,1e-5)+relative_mse(pt,tt,floor=tt)+3*relative_mse(pr,tr,floor=tt)+.25*gradient_relative_mse(pt,tt,tt)+.25*relative_mse(pg,g)+2*log_width_mse(pred,true)+log_peak_height_mse(pred,true,400,160000)
    loss.backward(); grad=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert pred.shape==(2,5) and pt.shape==(2,1000) and torch.isfinite(loss) and grad>0
    assert torch.all(pred[:,4]>0)
    print('EXP8 STRONG TEST PASSED'); print('parameter shape:',tuple(pred.shape)); print('curve shape:',tuple(pt.shape)); print('loss:',float(loss)); print('model parameters:',sum(p.numel() for p in model.parameters()))
if __name__=='__main__': main()
