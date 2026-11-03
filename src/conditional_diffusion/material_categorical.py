"""Matched optical conditioning and a fixed image-dependent categorical chain.

Rows of Q are old classes; columns are new classes. The reverse distribution is
the exact mixture of Bayes posteriors over the predicted clean class, not a
Gaussian DDIM update or the posterior with a soft label substituted into Bayes.
"""
import json
import torch
from torch import nn
from torch.nn import functional as F
from .static_semantic import StaticSemanticModel
from .material_conditions import load_direct


def grid(steps=5, total=100):
    if not 1<=steps<=total:raise ValueError('Invalid categorical steps')
    if steps==5 and total==100:return [100,75,50,25,1]
    return torch.linspace(total,1,steps).round().long().tolist()


def normalize(p):
    return p.clamp_min(1e-12)/p.clamp_min(1e-12).sum(-1,keepdim=True)


def categorical(p,uniform=None):
    p=normalize(p)
    if uniform is None:uniform=torch.rand(p.shape[:-1],device=p.device)
    # Four classes: explicit fixed-order sums avoid nondeterministic CUDA cumsum.
    first=p[...,0];second=first+p[...,1];third=second+p[...,2]
    thresholds=torch.stack([first,second,third],-1)
    return (uniform[...,None]>thresholds).sum(-1)


def label_channels(labels):return F.one_hot(labels,4).permute(0,3,1,2).float()


class MaterialModel(StaticSemanticModel):
    def __init__(self,variant,checkpoint=None,seed=42):
        if variant not in ['N0','N1','N2','N3']:raise ValueError(variant)
        super().__init__('S0' if variant=='N0' else 'S2',seed=seed,timesteps=100)
        self.variant=variant
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.condition_projection=nn.Conv2d(8,32,3,padding=1,bias=False)
            nn.init.zeros_(self.condition_projection.weight)
        if checkpoint is not None:
            mapped=load_direct(checkpoint).state_dict()
            destination=self.state_dict()
            for key,value in mapped.items():
                if key=='stage_timesteps':continue
                if key not in destination or destination[key].shape!=value.shape:raise ValueError('Incompatible warmstart '+key)
                destination[key]=value
            self.load_state_dict(destination,strict=True)

    def encode_conditions(self,image,conditions):
        if conditions.shape!=(len(image),8,*image.shape[-2:]):raise ValueError('Expected C3,r1,P4')
        encoder=self.rgb_encoder;h=encoder.input(image)+self.condition_projection(conditions)
        features=[]
        for i,blocks in enumerate(encoder.blocks):
            for block in blocks:h=block(h)
            features.append(h)
            if i<len(encoder.down):h=encoder.down[i](h)
        return tuple(features)

    def deterministic(self,image,conditions):
        features=self.encode_conditions(image,conditions)
        if self.variant=='N0':return self.forward_with_rgb(features)
        state=2*conditions[:,4:]-1
        for t in grid():
            output=self.forward_with_rgb(features,state,torch.full((len(image),),t,device=image.device,dtype=torch.long))
            state=2*output.softmax(1)-1
        return output


class CategoricalChain:
    def __init__(self,statistics,device='cpu',timesteps=100):
        s=json.loads(open(statistics).read()) if isinstance(statistics,(str,bytes)) else statistics
        self.total=timesteps
        self.mean=torch.tensor(s['mean'],device=device,dtype=torch.float32)
        cov=torch.tensor(s['covariance'],device=device,dtype=torch.float32)
        self.inverse=torch.linalg.inv(cov);self.logdet=torch.linalg.slogdet(cov)[1]
        self.similarity=torch.tensor(s['similarity'],device=device,dtype=torch.float32)
        self.global_relation=torch.tensor(s['global_relation'],device=device,dtype=torch.float32)

    @torch.no_grad()
    def relation(self,conditions,local=True):
        prior=normalize(conditions[:,4:].permute(0,2,3,1))
        if not local:return self.global_relation.expand(*prior.shape[:-1],4,4),prior
        color=conditions[:,:3].permute(0,2,3,1)
        delta=color[...,None,:]-self.mean
        quadratic=torch.einsum('...ki,kij,...kj->...k',delta,self.inverse,delta)
        evidence=(-.5*(quadratic+self.logdet)).softmax(-1)
        local_relation=normalize(self.similarity*evidence[...,None,:]+1e-4)
        reliability=conditions[:,3].clamp(0,1)[...,None,None]
        return reliability*local_relation+(1-reliability)*self.global_relation,prior

    def transition(self,relation,prior,t):
        # beta_t=1/(T-t+1), lambda_t=1-t/T. beta_T=1,lambda_T=0.
        time=torch.as_tensor(t,device=prior.device,dtype=torch.float32)
        if time.ndim==1:time=time[:,None,None,None,None]
        beta=1/(self.total-time+1);lam=1-time/self.total
        identity=torch.eye(4,device=prior.device,dtype=prior.dtype)
        return (1-beta)*identity+beta*(lam*relation+(1-lam)*prior[...,None,:])

    @torch.no_grad()
    def training_matrices(self,relation,prior,t):
        if t.shape!=(prior.shape[0],) or bool(((t<1)|(t>self.total)).any()):raise ValueError('t must be1..T')
        a=torch.eye(4,device=prior.device).expand(*prior.shape[:-1],4,4).clone();previous=a.clone()
        for step in range(1,int(t.max())+1):
            q=self.transition(relation,prior,step)
            advanced=normalize(a@q)
            a=torch.where((t>=step)[:,None,None,None,None],advanced,a)
            previous=torch.where((t>step)[:,None,None,None,None],a,previous)
        return previous,a,self.transition(relation,prior,t)

    @torch.no_grad()
    def inference_matrices(self,relation,prior,times):
        boundaries=sorted(set([0]+times))
        identity=torch.eye(4,device=prior.device).expand(*prior.shape[:-1],4,4)
        a=identity.clone();segment=identity.clone();prefix={0:identity};intervals={};start=0
        for step in range(1,max(times)+1):
            q=self.transition(relation,prior,step);a=normalize(a@q);segment=normalize(segment@q)
            if step in boundaries:
                prefix[step]=a.clone();intervals[(start,step)]=segment
                start=step;segment=identity.clone()
        return prefix,intervals

    @staticmethod
    def posterior(clean_probabilities,observed,previous,current,interval):
        """sum_y p_theta(y|x_t,I) q(x_s|x_t,y,I), including skipped steps."""
        column=interval.gather(-1,observed[...,None,None].expand(*observed.shape,4,1)).squeeze(-1)
        evidence=current.gather(-1,observed[...,None,None].expand(*observed.shape,4,1)).squeeze(-1)
        weights=clean_probabilities/evidence.clamp_min(1e-12)
        posterior=(weights.unsqueeze(-2)@previous).squeeze(-2)*column
        return normalize(posterior)

    def training_input(self,conditions,target,t,local):
        relation,prior=self.relation(conditions,local)
        previous,current,q=self.training_matrices(relation,prior,t)
        probabilities=current.gather(-2,target[...,None,None].expand(*target.shape,1,4)).squeeze(-2)
        observed=categorical(probabilities)
        true=self.posterior(F.one_hot(target,4).float(),observed,previous,current,q)
        return observed,true,(previous,current,q)


def classification_loss(logits,target):
    ce=F.cross_entropy(logits,target,reduction='none').mean()
    probs=logits.softmax(1);truth=F.one_hot(target,4).permute(0,3,1,2).float();dims=(0,2,3)
    dice=1-((2*(probs*truth).sum(dims)+1e-6)/(probs.sum(dims)+truth.sum(dims)+1e-6))[1:].mean()
    return .5*ce+.5*dice,dict(loss_ce=ce,loss_dice_fg=dice)


def train_step(model,chain,optimizer,image,conditions,target,config):
    optimizer.zero_grad(set_to_none=True)
    counts={}
    if model.variant in ['N0','N1']:
        output=model.deterministic(image,conditions);loss,terms=classification_loss(output,target)
    else:
        t=torch.randint(1,chain.total+1,(len(image),),device=image.device)
        observed,true,matrices=chain.training_input(conditions,target,t,model.variant=='N3')
        features=model.encode_conditions(image,conditions)
        output=model.forward_with_rgb(features,2*label_channels(observed)-1,t)
        loss,terms=classification_loss(output,target)
        predicted=chain.posterior(output.softmax(1).permute(0,2,3,1),observed,*matrices)
        kl=(true*(true.clamp_min(1e-12).log()-predicted.clamp_min(1e-12).log())).sum(-1).mean()
        loss=loss+config['kl_weight']*kl;terms['loss_posterior_kl']=kl
        counts={'timestep_sum':int(t.sum())}
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite material loss')
    loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
    optimizer.step();terms['loss_total']=loss
    return {k:float(v.detach()) for k,v in terms.items()},float(grad),counts


@torch.no_grad()
def sample(model,chain,image,conditions,uniforms=None,steps=5):
    if model.variant in ['N0','N1']:return model.deterministic(image,conditions).softmax(1)
    times=grid(steps,chain.total);features=model.encode_conditions(image,conditions)
    relation,prior=chain.relation(conditions,model.variant=='N3')
    prefix,intervals=chain.inference_matrices(relation,prior,times)
    observed=categorical(prior,None if uniforms is None else uniforms[:,0])
    for i,t in enumerate(times):
        output=model.forward_with_rgb(features,2*label_channels(observed)-1,
                                      torch.full((len(image),),t,device=image.device,dtype=torch.long))
        clean=output.softmax(1).permute(0,2,3,1)
        if i==len(times)-1:return clean.permute(0,3,1,2)
        earlier=times[i+1]
        posterior=chain.posterior(clean,observed,prefix[earlier],prefix[t],intervals[(earlier,t)])
        observed=categorical(posterior,None if uniforms is None else uniforms[:,i+1])
    raise RuntimeError('Empty categorical schedule')
