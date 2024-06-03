import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
   
   def __init__(self, train_posn, train_meas, likelihood):
      super(ExactGPModel, self).__init__(train_posn, train_meas, likelihood)  
      self.mean_module  = gpytorch.means.ConstantMean()
      self.base_kernel  = gpytorch.kernels.RBFKernel(ard_num_dims = 2)
      self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
      
      hypers = {'likelihood.noise_covar.noise': torch.tensor(0.1),
               'covar_module.base_kernel.lengthscale': torch.tensor([10, 10]),
               'covar_module.outputscale': torch.tensor(1.)}
      self.initialize(**hypers)
      
   def forward(self, x):
      mean_x = self.mean_module(x)
      covar_x = self.covar_module(x)
      return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
   

def mtrain(model, mll):
   optimizer = torch.optim.Rprop(model.parameters(), lr=0.1)
   best_loss = float('inf')
   patience  = 10
   
   hypers = {'likelihood.noise_covar.noise': torch.tensor(0.1),
             'covar_module.base_kernel.lengthscale': torch.tensor([10, 10]),
             'covar_module.outputscale': torch.tensor(1.)}
   
   for i in range(100):
      optimizer.zero_grad()
      output = model(model.train_inputs[0])
      loss   =-mll(output, model.train_targets)     
      loss.backward()
      print('Iter %d - Loss: %.3f    noise: %.3f' % (i + 1, loss.item(), model.likelihood.noise.item()))
      optimizer.step()
      val_loss = loss.item()
      
      if val_loss < 0.99*best_loss:
         best_loss = val_loss
         hypers["likelihood.noise_covar.noise"] = model.likelihood.noise_covar.noise.item()
         hypers["covar_module.base_kernel.lengthscale"] = model.covar_module.base_kernel.lengthscale[0]
         hypers["covar_module.outputscale"] = model.covar_module.outputscale.item()
         patience = 10  # Reset patience counter
      else:
         patience -= 1
         if patience == 0:
            lengthscale = model.covar_module.base_kernel.lengthscale[0]
            print('Best_Loss: %.3f  noise: %.3f lengthscale: %0.3f %0.3f' % (best_loss, hypers["likelihood.noise_covar.noise"], lengthscale[0], lengthscale[1]))
            model.initialize(**hypers)
            break    
     
def measure(model, likelihood, loca):
   model.eval()
   likelihood.eval()
   with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
      predictions = likelihood(model(loca))
      return predictions.mean
   
  
