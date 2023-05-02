"""Custom policy classes and convenience methods."""

import abc
from typing import Type

import gym
import numpy as np
import torch as th
from stable_baselines3.common import policies, torch_layers
from stable_baselines3.sac import policies as sac_policies
from torch import nn

from imitation.util import networks


class HardCodedPolicy(policies.BasePolicy, abc.ABC):
    """Abstract class for hard-coded (non-trainable) policies."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """Builds HardcodedPolicy with specified observation and action space."""
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )

    def _predict(self, obs: th.Tensor, deterministic: bool = False):
        np_actions = []
        np_obs = obs.detach().cpu().numpy()
        for np_ob in np_obs:
            assert self.observation_space.contains(np_ob)
            np_actions.append(self._choose_action(np_ob))
        np_actions = np.stack(np_actions, axis=0)
        th_actions = th.as_tensor(np_actions, device=self.device)
        return th_actions

    @abc.abstractmethod
    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        """Chooses an action, optionally based on observation obs."""

    def forward(self, *args):
        # technically BasePolicy is a Torch module, so this needs a forward()
        # method
        raise NotImplementedError  # pragma: no cover


class RandomPolicy(HardCodedPolicy):
    """Returns random actions."""

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return self.action_space.sample()


class ZeroPolicy(HardCodedPolicy):
    """Returns constant zero action."""

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_space.shape, dtype=self.action_space.dtype)


class FeedForward32Policy(policies.ActorCriticPolicy):
    """A feed forward policy network with two hidden layers of 32 units.

    This matches the IRL policies in the original AIRL paper.

    Note: This differs from stable_baselines3 ActorCriticPolicy in two ways: by
    having 32 rather than 64 units, and by having policy and value networks
    share weights except at the final layer, where there are different linear heads.
    """

    def __init__(self, *args, **kwargs):
        """Builds FeedForward32Policy; arguments passed to `ActorCriticPolicy`."""
        super().__init__(*args, **kwargs, net_arch=[32, 32])

class FFLayer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.leakyrelu = torch.nn.LeakyReLU()
        self.rrelu = torch.nn.RReLU()
        self.gelu = torch.nn.GELU()
        self.opt = torch.optim.AdamW(self.parameters(), lr=0.02)
        self.threshold = 2.0

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        g_pos = self.forward(x_pos).pow(2).mean(1)
        g_neg = self.forward(x_neg).pow(2).mean(1)
        # The following loss pushes pos (neg) samples to values larger (smaller) than the self.threshold.
        loss = torch.log(1 + torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold]))).mean()
        self.opt.zero_grad()
        # this backward just compute the derivative and hence is not considered backpropagation.
        loss.backward()
        self.opt.step()
        return (self.forward(x_pos).detach()+self.forward(x_neg).detach())/2

class MlpExtractor(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(FFLayer(last_layer_dim_pi, curr_layer_dim))
            
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(FFLayer(last_layer_dim_vf, curr_layer_dim))
            

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = policy_net.to(device)
        self.value_net = value_net.to(device)

    def forward(self,  features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        pos , neg = self.create_data(features)
        return self.forward_actor(pos,neg), self.forward_critic(pos,neg)

    def forward_actor(self, pos: th.Tensor , neg: th.Tensor) -> th.Tensor:
        return self.train(pos,neg)

    def forward_critic(self, pos: th.Tensor , neg: th.Tensor) -> th.Tensor:
        return self.train(pos,neg)
    
    def create_data(self, features):
      pass 
      ## This function will be updated after pseudo reward model is trained.

class ForwardForward32Policy(policies.ActorCriticPolicy):
    

    def __init__(self, *args, **kwargs):
        """Builds ForwardForward32Policy; arguments passed to `ActorCriticPolicy`."""
        super().__init__(*args, **kwargs, net_arch=[32, 32])

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob
    
    




class SAC1024Policy(sac_policies.SACPolicy):
    """Actor and value networks with two hidden layers of 1024 units respectively.

    This matches the implementation of SAC policies in the PEBBLE paper. See:
    https://arxiv.org/pdf/2106.05091.pdf
    https://github.com/denisyarats/pytorch_sac/blob/master/config/agent/sac.yaml

    Note: This differs from stable_baselines3 SACPolicy by having 1024 hidden units
    in each layer instead of the default value of 256.
    """

    def __init__(self, *args, **kwargs):
        """Builds SAC1024Policy; arguments passed to `SACPolicy`."""
        super().__init__(*args, **kwargs, net_arch=[1024, 1024])


class NormalizeFeaturesExtractor(torch_layers.FlattenExtractor):
    """Feature extractor that flattens then normalizes input."""

    def __init__(
        self,
        observation_space: gym.Space,
        normalize_class: Type[nn.Module] = networks.RunningNorm,
    ):
        """Builds NormalizeFeaturesExtractor.

        Args:
            observation_space: The space observations lie in.
            normalize_class: The class to use to normalize observations (after being
                flattened). This can be any Module that preserves the shape;
                e.g. `nn.BatchNorm*` or `nn.LayerNorm`.
        """
        super().__init__(observation_space)
        # Below we have to ignore the type error when initializing the class because
        # there is no simple way of specifying a protocol that admits one positional
        # argument for the number of features while being compatible with nn.Module.
        # (it would require defining a base class and forcing all the subclasses
        # to inherit from it).
        self.normalize = normalize_class(self.features_dim)  # type: ignore[call-arg]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        flattened = super().forward(observations)
        return self.normalize(flattened)
