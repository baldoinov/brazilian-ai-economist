import os
from gymnasium import Space
import numpy as np

from gymnasium.spaces import Box
from gymnasium.spaces import Dict
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.utils.framework import try_import_tf


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf = try_import_tf()

def get_flat_obs_size(obs_space):
    if isinstance(obs_space, Box):
        return np.prod(obs_space.shape)
    elif not isinstance(obs_space, Dict):
        raise TypeError

    def rec_size(obs_dict_space, n=0):
        for subspace in obs_dict_space.spaces.values():
            if isinstance(subspace, Box):
                n = n + np.prod(subspace.shape)
            elif isinstance(subspace, Dict):
                n = rec_size(subspace, n=n)
            else:
                raise TypeError
        return n

    return rec_size(obs_space)


def apply_logit_mask(logits, mask):
    """Mask values of 1 are valid actions."
    " Add huge negative values to logits with 0 mask values."""
    logit_mask = tf.ones_like(logits) * -10000000
    logit_mask = logit_mask * (1 - mask)

    return logits + logit_mask


_WORLD_MAP_NAME = "world-map"
_WORLD_IDX_MAP_NAME = "world-idx_map"
_MASK_NAME = "action_mask"


class KerasConvolutionalLSTM(TFModelV2):
    custom_name = "KerasConvolutionalLSTM"

    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int, model_config: dict, name: str):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        input_emb_vocab = self.model_config["custom_options"]["input_emb_vocab"]
        emb_dim = self.model_config["custom_options"]["idx_emb_dim"]
        num_conv = self.model_config["custom_options"]["num_conv"]
        num_fc = self.model_config["custom_options"]["num_fc"]
        fc_dim = self.model_config["custom_options"]["fc_dim"]
        cell_size = self.model_config["custom_options"]["lstm_cell_size"]
        generic_name = self.model_config["custom_options"].get("generic_name", None)

        self.cell_size = cell_size

        if hasattr(obs_space, "original_space"):
            obs_space = obs_space.original_space

        if not isinstance(obs_space, Dict):
            if isinstance(obs_space, Box):
                raise TypeError(
                    "({}) Observation space should be a gym Dict."
                    " Is a Box of shape {}".format(name, obs_space.shape)
                )
            raise TypeError(
                "({}) Observation space should be a gym Dict."
                " Is {} instead.".format(name, type(obs_space))
            )

        # Define input layers
        self._input_keys = []
        non_conv_input_keys = []
        input_dict = {}
        conv_shape_r = None
        conv_shape_c = None
        conv_map_channels = None
        conv_idx_channels = None
        found_world_map = False
        found_world_idx = False
        for k, v in obs_space.spaces.items():
            shape = (None,) + v.shape
            input_dict[k] = tf.keras.layers.Input(shape=shape, name=k)
            self._input_keys.append(k)
            if k == _MASK_NAME:
                pass
            elif k == _WORLD_MAP_NAME:
                conv_shape_r, conv_shape_c, conv_map_channels = (
                    v.shape[1],
                    v.shape[2],
                    v.shape[0],
                )
                found_world_map = True
            elif k == _WORLD_IDX_MAP_NAME:
                conv_idx_channels = v.shape[0] * emb_dim
                found_world_idx = True
            else:
                non_conv_input_keys.append(k)

        # Cell state and hidden state for the
        # policy and value function networks.
        state_in_h_p = tf.keras.layers.Input(shape=(cell_size,), name="h_pol")
        state_in_c_p = tf.keras.layers.Input(shape=(cell_size,), name="c_pol")
        state_in_h_v = tf.keras.layers.Input(shape=(cell_size,), name="h_val")
        state_in_c_v = tf.keras.layers.Input(shape=(cell_size,), name="c_val")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in")

        # Determine which of the inputs are treated as non-conv inputs
        if generic_name is None:
            non_conv_inputs = tf.keras.layers.concatenate(
                [input_dict[k] for k in non_conv_input_keys]
            )
        elif isinstance(generic_name, (tuple, list)):
            non_conv_inputs = tf.keras.layers.concatenate(
                [input_dict[k] for k in generic_name]
            )
        elif isinstance(generic_name, str):
            non_conv_inputs = input_dict[generic_name]
        else:
            raise TypeError

        if found_world_map:
            assert found_world_idx
            use_conv = True
            conv_shape = (
                conv_shape_r,
                conv_shape_c,
                conv_map_channels + conv_idx_channels,
            )

            conv_input_map = tf.keras.layers.Permute((1, 3, 4, 2))(
                input_dict[_WORLD_MAP_NAME]
            )
            conv_input_idx = tf.keras.layers.Permute((1, 3, 4, 2))(
                input_dict[_WORLD_IDX_MAP_NAME]
            )

        else:
            assert not found_world_idx
            use_conv = False
            conv_shape = None
            conv_input_map = None
            conv_input_idx = None

        logits, values, state_h_p, state_c_p, state_h_v, state_c_v = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

        # Define the policy and value function models
        for tag in ["_pol", "_val"]:
            if tag == "_pol":
                state_in = [state_in_h_p, state_in_c_p]
            elif tag == "_val":
                state_in = [state_in_h_v, state_in_c_v]
            else:
                raise NotImplementedError

            # Apply convolution to the spatial inputs
            if use_conv:
                map_embedding = tf.keras.layers.Embedding(
                    input_emb_vocab, emb_dim, name="embedding" + tag
                )
                conv_idx_embedding = tf.keras.layers.Reshape(
                    (-1, conv_shape_r, conv_shape_c, conv_idx_channels)
                )(map_embedding(conv_input_idx))

                conv_input = tf.keras.layers.concatenate(
                    [conv_input_map, conv_idx_embedding]
                )

                conv_model = tf.keras.models.Sequential(name="conv_model" + tag)
                assert conv_shape
                conv_model.add(
                    tf.keras.layers.Conv2D(
                        16,
                        (3, 3),
                        strides=2,
                        activation="relu",
                        input_shape=conv_shape,
                        name="conv2D_1" + tag,
                    )
                )

                for i in range(num_conv - 1):
                    conv_model.add(
                        tf.keras.layers.Conv2D(
                            32,
                            (3, 3),
                            strides=2,
                            activation="relu",
                            name="conv2D_{}{}".format(i + 2, tag),
                        )
                    )

                conv_model.add(tf.keras.layers.Flatten())

                conv_td = tf.keras.layers.TimeDistributed(conv_model)(conv_input)

                # Combine the conv output with the non-conv inputs
                dense = tf.keras.layers.concatenate([conv_td, non_conv_inputs])

            # No spatial inputs provided -- skip any conv steps
            else:
                dense = non_conv_inputs

            # Preprocess observation with hidden layers and send to LSTM cell
            for i in range(num_fc):
                layer = tf.keras.layers.Dense(
                    fc_dim, activation=tf.nn.relu, name="dense{}".format(i + 1) + tag
                )
                dense = layer(dense)

            dense = tf.keras.layers.LayerNormalization(name="layer_norm" + tag)(dense)

            lstm_out, state_h, state_c = tf.keras.layers.LSTM(
                cell_size, return_sequences=True, return_state=True, name="lstm" + tag
            )(inputs=dense, mask=tf.sequence_mask(seq_in), initial_state=state_in)

            # Project LSTM output to logits or value
            output = tf.keras.layers.Dense(
                self.num_outputs if tag == "_pol" else 1,
                activation=tf.keras.activations.linear,
                name="logits" if tag == "_pol" else "value",
            )(lstm_out)

            if tag == "_pol":
                state_h_p, state_c_p = state_h, state_c
                logits = apply_logit_mask(output, input_dict[_MASK_NAME])
            elif tag == "_val":
                state_h_v, state_c_v = state_h, state_c
                values = output
            else:
                raise NotImplementedError

        self.input_dict = input_dict

        # This will be set in the forward_rnn() call below
        self._value_out = None

        for out in [logits, values, state_h_p, state_c_p, state_h_v, state_c_v]:
            assert out is not None

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=self._extract_input_list(input_dict)
            + [seq_in, state_in_h_p, state_in_c_p, state_in_h_v, state_in_c_v],
            outputs=[logits, values, state_h_p, state_c_p, state_h_v, state_c_v],
        )
        self.register_variables(self.rnn_model.variables)
        # self.rnn_model.summary()

    def _extract_input_list(self, dictionary):
        return [dictionary[k] for k in self._input_keys]

    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        output, new_state = self.forward_rnn(
            [
                add_time_dimension(t, seq_lens)
                for t in self._extract_input_list(input_dict["obs"])
            ],
            state,
            seq_lens,
        )
        return tf.reshape(output, [-1, self.num_outputs]), new_state

    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h_p, c_p, h_v, c_v = self.rnn_model(
            inputs + [seq_lens] + state
        )
        return model_out, [h_p, c_p, h_v, c_v]

    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


ModelCatalog.register_custom_model(KerasConvolutionalLSTM.custom_name, KerasConvolutionalLSTM)

