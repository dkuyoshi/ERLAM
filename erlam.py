import copy
from logging import getLogger

import chainer
from chainer import cuda
import chainer.functions as F

from chainerrl import agent
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.replay_buffer import batch_experiences
from chainerrl.replay_buffer import batch_recurrent_experiences
from chainerrl.replay_buffer import ReplayUpdater

from associative_memory import AssociativeMemory

import networkx as nx


def compute_value_loss(y, t, clip_delta=True, batch_accumulator='mean'):
    """Compute a loss for value prediction problem.

    Args:
        y (Variable or ndarray): Predicted values.
        t (Variable or ndarray): Target values.
        clip_delta (bool): Use the Huber loss function if set True.
        batch_accumulator (str): 'mean' or 'sum'. 'mean' will use the mean of
            the loss values in a batch. 'sum' will use the sum.
    Returns:
        (Variable) scalar loss
    """
    assert batch_accumulator in ('mean', 'sum')
    y = F.reshape(y, (-1, 1))
    t = F.reshape(t, (-1, 1))
    if clip_delta:
        loss_sum = F.sum(F.huber_loss(y, t, delta=1.0))
        if batch_accumulator == 'mean':
            loss = loss_sum / y.shape[0]
        elif batch_accumulator == 'sum':
            loss = loss_sum
    else:
        loss_mean = F.mean_squared_error(y, t) / 2
        if batch_accumulator == 'mean':
            loss = loss_mean
        elif batch_accumulator == 'sum':
            loss = loss_mean * y.shape[0]
    return loss


def compute_value_loss_erlam(y, t, qg, lambdas, clip_delta=True, batch_accumulator='mean'):
    """Compute a loss for value prediction problem. --- ERLAM ver

    Args:
        y (Variable or ndarray): Predicted values.
        t (Variable or ndarray): Target values.
        clip_delta (bool): Use the Huber loss function if set True.
        batch_accumulator (str): 'mean' or 'sum'. 'mean' will use the mean of
            the loss values in a batch. 'sum' will use the sum.
    Returns:
        (Variable) scalar loss
    """
    assert batch_accumulator in ('mean', 'sum')
    y = F.reshape(y, (-1, 1))
    t = F.reshape(t, (-1, 1))
    qg = F.reshape(qg, (-1, 1))
    if clip_delta:
        loss_sum = F.sum(F.huber_loss(y, t, delta=1.0)) + lambdas * F.sum(F.huber_loss(y, qg, delta=1.0))
        if batch_accumulator == 'mean':
            loss = loss_sum / y.shape[0]
        elif batch_accumulator == 'sum':
            loss = loss_sum
    else:
        loss_mean = (F.mean_squared_error(y, t) + lambdas * F.mean_squared_error(qg, y)) / 2
        if batch_accumulator == 'mean':
            loss = loss_mean
        elif batch_accumulator == 'sum':
            loss = loss_mean * y.shape[0]
    return loss


def compute_weighted_value_loss(y, t, weights,
                                clip_delta=True, batch_accumulator='mean'):
    """Compute a loss for value prediction problem.

    Args:
        y (Variable or ndarray): Predicted values.
        t (Variable or ndarray): Target values.
        weights (ndarray): Weights for y, t.
        clip_delta (bool): Use the Huber loss function if set True.
        batch_accumulator (str): 'mean' will divide loss by batchsize
    Returns:
        (Variable) scalar loss
    """
    assert batch_accumulator in ('mean', 'sum')
    y = F.reshape(y, (-1, 1))
    t = F.reshape(t, (-1, 1))
    if clip_delta:
        losses = F.huber_loss(y, t, delta=1.0)
    else:
        losses = F.square(y - t) / 2
    losses = F.reshape(losses, (-1,))
    loss_sum = F.sum(losses * weights)
    if batch_accumulator == 'mean':
        loss = loss_sum / y.shape[0]
    elif batch_accumulator == 'sum':
        loss = loss_sum
    return loss


def _batch_reset_recurrent_states_when_episodes_end(
        model, batch_done, batch_reset, recurrent_states):
    """Reset recurrent states when episodes end.

    Args:
        model (chainer.Link): Model that implements `StatelessRecurrent`.
        batch_done (array-like of bool): True iff episodes are terminal.
        batch_reset (array-like of bool): True iff episodes will be reset.
        recurrent_states (object): Recurrent state.

    Returns:
        object: New recurrent states.
    """
    indices_that_ended = [
        i for i, (done, reset)
        in enumerate(zip(batch_done, batch_reset)) if done or reset]
    if indices_that_ended:
        return model.mask_recurrent_state_at(
            recurrent_states, indices_that_ended)
    else:
        return recurrent_states


class ERLAM(agent.AttributeSavingMixin, agent.BatchAgent):
    """EPISODIC REINFORCEMENT LEARNING WITH ASSOCIATIVE MEMORY.

    Args:
        q_function (StateQFunction): Q-function
        optimizer (Optimizer): Optimizer that is already setup
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        k (int): number of K
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        target_update_interval (int): Target model update interval in step
        clip_delta (bool): Clip delta if set True
        phi (callable): Feature extractor applied to observations
        target_update_method (str): 'hard' or 'soft'.
        soft_update_tau (float): Tau of soft target update.
        n_times_update (int): Number of repetition of update
        average_q_decay (float): Decay rate of average Q, only used for
            recording statistics
        average_loss_decay (float): Decay rate of average loss, only used for
            recording statistics
        batch_accumulator (str): 'mean' or 'sum'
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
        recurrent (bool): If set to True, `model` is assumed to implement
            `chainerrl.links.StatelessRecurrent` and is updated in a recurrent
            manner.
    """
    episode_idx: int

    saved_attributes = ('model', 'target_model', 'optimizer')

    def __init__(self, q_function, optimizer, replay_buffer, gamma, k,
                 lambdas, dim, start_size, capacity,
                 explorer, gpu=None, replay_start_size=50000,
                 minibatch_size=3, update_interval=1,
                 target_update_interval=10000, clip_delta=True,
                 phi=lambda x: x,
                 target_update_method='hard',
                 soft_update_tau=1e-2,
                 n_times_update=1, average_q_decay=0.999,
                 average_loss_decay=0.99,
                 batch_accumulator='mean',
                 episodic_update_len=None,
                 logger=getLogger(__name__),
                 batch_states=batch_states,
                 recurrent=False,
                 ):
        self.model = q_function
        self.q_function = q_function  # For backward compatibility

        if gpu is not None and gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            self.model.to_gpu(device=gpu)

        self.xp = self.model.xp
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.clip_delta = clip_delta
        self.phi = phi
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.batch_accumulator = batch_accumulator
        assert batch_accumulator in ('mean', 'sum')
        self.logger = logger
        self.batch_states = batch_states
        self.recurrent = recurrent
        if self.recurrent:
            update_func = self.update_from_episodes
        else:
            update_func = self.update
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=update_func,
            batchsize=minibatch_size,
            episodic_update=recurrent,
            episodic_update_len=episodic_update_len,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )

        self.t = 0
        self.last_state = None
        self.last_action = None
        self.target_model = None
        self.last_hidden_vector = None
        self.sync_target_network()
        # For backward compatibility
        self.target_q_function = self.target_model
        self.average_q = 0
        self.average_q_decay = average_q_decay
        self.average_loss = 0
        self.average_loss_decay = average_loss_decay

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.train_prev_recurrent_states = None
        self.test_recurrent_states = None

        # Associative Memoryの定義(グラフ)
        self.associative_memory = AssociativeMemory(capacity=capacity, xp=self.xp, dim=dim)

        # 逆順で追加していくやつのためのリスト定義
        self.embeddings_back = []
        self.actions_back = []
        self.rewards_back = []
        self.id_backs = []

        # episode numberたちもここで管理
        self.te = 0
        self.episode_idx = 0
        self.k = k

        # その他ハイパーパラメータ
        self.lambdas = lambdas
        self.start_size = start_size

        # Error checking
        if (self.replay_buffer.capacity is not None and
                self.replay_buffer.capacity <
                self.replay_updater.replay_start_size):
            raise ValueError(
                'Replay start size cannot exceed '
                'replay buffer capacity.')

    def sync_target_network(self):
        """Synchronize target network with current network."""
        if self.target_model is None:
            self.target_model = copy.deepcopy(self.model)
            call_orig = self.target_model.__call__

            def call_test(self_, x):
                with chainer.using_config('train', False):
                    return call_orig(self_, x)

            self.target_model.__call__ = call_test
        else:
            synchronize_parameters(
                src=self.model,
                dst=self.target_model,
                method=self.target_update_method,
                tau=self.soft_update_tau)

    def update(self, experiences, errors_out=None):
        """Update the model from experiences in ERLAM

        Args:
            experiences (list): List of lists of dicts.
                For DQN, each dict must contains:
                  - state (object): State
                  - action (object): Action
                  - reward (float): Reward
                  - is_state_terminal (bool): True iff next state is terminal
                  - next_state (object): Next state
                  - weight (float, optional): Weight coefficient. It can be
                    used for importance sampling.
            errors_out (list or None): If set to a list, then TD-errors
                computed from the given experiences are appended to the list.

        Returns:
            None
        """
        has_weight = 'weight' in experiences[0][0]
        exp_batch = batch_experiences(
            experiences, xp=self.xp,
            phi=self.phi, gamma=self.gamma,
            batch_states=self.batch_states)
        if has_weight:
            exp_batch['weights'] = self.xp.asarray(
                [elem[0]['weight'] for elem in experiences],
                dtype=self.xp.float32)
            if errors_out is None:
                errors_out = []
        loss = self._compute_loss(exp_batch, errors_out=errors_out)

        if has_weight:
            self.replay_buffer.update_errors(errors_out)

        # Update stats
        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * float(loss.array)

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()

    def update_from_episodes(self, episodes, errors_out=None):
        assert errors_out is None, \
            "Recurrent DQN does not support PrioritizedBuffer"
        exp_batch = batch_recurrent_experiences(
            episodes,
            model=self.model,
            xp=self.xp,
            phi=self.phi, gamma=self.gamma,
            batch_states=self.batch_states,
        )
        loss = self._compute_loss(exp_batch, errors_out=None)
        # Update stats
        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * float(loss.array)
        self.optimizer.update(lambda: loss)

    def _compute_target_values(self, exp_batch):
        batch_next_state = exp_batch['next_state']

        if self.recurrent:
            target_next_qout, _ = self.target_model.n_step_forward(
                batch_next_state, exp_batch['next_recurrent_state'],
                output_mode='concat')
        else:
            target_next_qout = self.target_model(batch_next_state)
        next_q_max = target_next_qout.max

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']
        discount = exp_batch['discount']

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q_max

    def _compute_y_and_t(self, exp_batch):
        batch_size = exp_batch['reward'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        if self.recurrent:
            qout, _ = self.model.n_step_forward(
                batch_state,
                exp_batch['recurrent_state'],
                output_mode='concat',
            )
        else:
            qout = self.model(batch_state)

        batch_actions = exp_batch['action']
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        with chainer.no_backprop_mode():
            batch_q_target = F.reshape(
                self._compute_target_values(exp_batch),
                (batch_size, 1))

        return batch_q, batch_q_target

    def _compute_loss(self, exp_batch, errors_out=None):
        """Compute the Q-learning loss for a batch of experiences

        Args:
          exp_batch (dict): A dict of batched arrays of transitions
        Returns:
          Computed loss from the minibatch of experiences
        """
        y, t = self._compute_y_and_t(exp_batch)

        if errors_out is not None:
            del errors_out[:]
            delta = F.absolute(y - t)
            if delta.ndim == 2:
                delta = F.sum(delta, axis=1)
            delta = cuda.to_cpu(delta.array)
            for e in delta:
                errors_out.append(e)

        if 'weights' in exp_batch:
            return compute_weighted_value_loss(
                y, t, exp_batch['weights'],
                clip_delta=self.clip_delta,
                batch_accumulator=self.batch_accumulator)
        else:
            if nx.number_of_nodes(self.associative_memory.graph) >= self.start_size:
                # GraphからのLoss計算
                batch_h = self.model.batch_get_embedding_each_frame(exp_batch['state'])
                qg = self.associative_memory.get_q(batch_h, exp_batch['action'])
                return compute_value_loss_erlam(y, t, qg, self.lambdas, clip_delta=self.clip_delta,
                                                batch_accumulator=self.batch_accumulator)
            else:
                return compute_value_loss(y, t, clip_delta=self.clip_delta,
                                          batch_accumulator=self.batch_accumulator)

    def act(self, obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value, _ = \
                self._evaluate_model_and_update_recurrent_states(
                    [obs], test=True)
            q = float(action_value.max.array)
            action = cuda.to_cpu(action_value.greedy_actions.array)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)
        return action

    def act_and_train(self, obs, reward):

        # Observe the consequences
        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer
            transition = {
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'next_state': obs,
                'is_state_terminal': False,
            }

            if self.recurrent:
                transition['recurrent_state'] = \
                    self.model.get_recurrent_state_at(
                        self.train_prev_recurrent_states,
                        0, unwrap_variable=True)
                self.train_prev_recurrent_states = None
                transition['next_recurrent_state'] = \
                    self.model.get_recurrent_state_at(
                        self.train_recurrent_states, 0, unwrap_variable=True)
            self.replay_buffer.append(**transition)

        # Update the target network
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

        # Update the model
        self.replay_updater.update_if_necessary(self.t)

        # Choose an action
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value, hidden_vector = \
                self._evaluate_model_and_update_recurrent_states(
                    [obs], test=False)
            q = float(action_value.max.array)
            greedy_action = cuda.to_cpu(action_value.greedy_actions.array)[0]
        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        # Add information for Backing Trajectory
        if self.last_state is not None:
            assert self.last_hidden_vector is not None
            self.embeddings_back.append(self.last_hidden_vector)
            self.actions_back.append(self.last_action)
            self.rewards_back.append(reward)
            self.id_backs.append(self.te)
            # This is ID
            self.te += 1

        self.t += 1
        self.last_state = obs
        self.last_action = action
        self.last_hidden_vector = hidden_vector

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)
        self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        return self.last_action

    def _evaluate_model_and_update_recurrent_states(self, batch_obs, test):
        batch_xs = self.batch_states(batch_obs, self.xp, self.phi)
        if self.recurrent:
            if test:
                batch_av, self.test_recurrent_states = self.model(
                    batch_xs, self.test_recurrent_states)
            else:
                self.train_prev_recurrent_states = self.train_recurrent_states
                batch_av, self.train_recurrent_states = self.model(
                    batch_xs, self.train_recurrent_states)
        else:
            batch_av = self.model(batch_xs)
            # batch_h = self.model.get_embedding(batch_xs)
            batch_h16 = self.model.get_embedding_each_frame(batch_xs)
        return batch_av, batch_h16

    def batch_act_and_train(self, batch_obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            batch_av = self._evaluate_model_and_update_recurrent_states(
                batch_obs, test=False)
            batch_maxq = batch_av.max.array
            batch_argmax = cuda.to_cpu(batch_av.greedy_actions.array)
        batch_action = [
            self.explorer.select_action(
                self.t, lambda: batch_argmax[i],
                action_value=batch_av[i:i + 1],
            )
            for i in range(len(batch_obs))]
        self.batch_last_obs = list(batch_obs)
        self.batch_last_action = list(batch_action)

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * float(batch_maxq.mean())

        return batch_action

    def batch_act(self, batch_obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            batch_av = self._evaluate_model_and_update_recurrent_states(
                batch_obs, test=True)
            batch_argmax = cuda.to_cpu(batch_av.greedy_actions.array)
            return batch_argmax

    def batch_observe_and_train(self, batch_obs, batch_reward,
                                batch_done, batch_reset):
        for i in range(len(batch_obs)):
            self.t += 1
            # Update the target network
            if self.t % self.target_update_interval == 0:
                self.sync_target_network()
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                transition = {
                    'state': self.batch_last_obs[i],
                    'action': self.batch_last_action[i],
                    'reward': batch_reward[i],
                    'next_state': batch_obs[i],
                    'next_action': None,
                    'is_state_terminal': batch_done[i],
                }
                if self.recurrent:
                    transition['recurrent_state'] = \
                        self.model.get_recurrent_state_at(
                            self.train_prev_recurrent_states,
                            i, unwrap_variable=True)
                    transition['next_recurrent_state'] = \
                        self.model.get_recurrent_state_at(
                            self.train_recurrent_states,
                            i, unwrap_variable=True)
                self.replay_buffer.append(env_id=i, **transition)
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)

        if self.recurrent:
            # Reset recurrent states when episodes end
            self.train_prev_recurrent_states = None
            self.train_recurrent_states = \
                _batch_reset_recurrent_states_when_episodes_end(
                    model=self.model,
                    batch_done=batch_done,
                    batch_reset=batch_reset,
                    recurrent_states=self.train_recurrent_states,
                )

    def batch_observe(self, batch_obs, batch_reward,
                      batch_done, batch_reset):
        if self.recurrent:
            # Reset recurrent states when episodes end
            self.test_recurrent_states = \
                _batch_reset_recurrent_states_when_episodes_end(
                    model=self.model,
                    batch_done=batch_done,
                    batch_reset=batch_reset,
                    recurrent_states=self.test_recurrent_states,
                )

    def stop_episode_and_train(self, state, reward, done=False):
        """Observe a terminal state and a reward.

        This function must be called once when an episode terminates.
        """

        assert self.last_state is not None
        assert self.last_action is not None

        # Add a transition to the replay buffer
        transition = {
            'state': self.last_state,
            'action': self.last_action,
            'reward': reward,
            'next_state': state,
            'next_action': self.last_action,
            'is_state_terminal': done,
        }
        if self.recurrent:
            transition['recurrent_state'] = \
                self.model.get_recurrent_state_at(
                    self.train_prev_recurrent_states, 0, unwrap_variable=True)
            self.train_prev_recurrent_states = None
            transition['next_recurrent_state'] = \
                self.model.get_recurrent_state_at(
                    self.train_recurrent_states, 0, unwrap_variable=True)
        self.replay_buffer.append(**transition)

        self.embeddings_back.append(self.last_hidden_vector)
        self.actions_back.append(self.last_action)
        self.rewards_back.append(reward)
        self.id_backs.append(self.te)

        # Rt = []
        embeddings_back = self.embeddings_back[::-1]
        actions_back = self.actions_back[::-1]
        rewards_back = self.rewards_back[::-1]
        id_backs = self.id_backs[::-1]

        assert len(rewards_back) == len(id_backs) and len(id_backs) == len(embeddings_back), '何かがおかしい'

        # グラフに追加していく(for t=Te...1) and Rtの計算も含めて
        initial_step = True
        '''
        for embedding, action, reward, t in zip(embeddings_back, actions_back, rewards_back, id_backs):
            if initial_step:
                Rt_history = reward
                # Rt.append(Rt_history)
                Rt = Rt_history
                # self.associative_memory.append(embedding, action, reward, t, Rt)
                initial_step = False
            else:
                Rt_history = reward + self.gamma * Rt_history
                # Rt.append(Rt_history)
                Rt = Rt_history
                # self.associative_memory.append(embedding, action, reward, t, Rt)
        '''
        Rt = []
        for reward in rewards_back:
            if initial_step:
                Rt_history = reward
                Rt.append(Rt_history)
                # self.associative_memory.append(embedding, action, reward, t, Rt)
                initial_step = False
            else:
                Rt_history = reward + self.gamma * Rt_history
                Rt.append(Rt_history)
                # self.associative_memory.append(embedding, action, reward, t, Rt)

        self.associative_memory.append_collectively(embeddings_back, actions_back, rewards_back, id_backs, Rt)
        self.associative_memory.add_edge()
        # self.associative_memory.visualize_graph()
        # self.associative_memory.append_collectively(embeddings_back, actions_back, rewards_back, id_backs, Rt)

        # それぞれ初期化
        self._back_information_reset()
        self.episode_idx += 1

        # if episode_idx % K == 0 -> Algorithm 1
        if self.episode_idx % self.k == 0:
            self._value_propagation()

        self.last_hidden_vector = None
        self.last_state = None
        self.last_action = None

        if self.recurrent:
            self.train_recurrent_states = None

        self.replay_buffer.stop_current_episode()

    def _value_propagation(self):
        self.associative_memory.value_propagation(self.gamma)

    def _back_information_reset(self):
        self.embeddings_back = []
        self.actions_back = []
        self.rewards_back = []
        self.id_backs = []
        self.te = 0

    def stop_episode(self):
        if self.recurrent:
            self.test_recurrent_states = None

    def get_statistics(self):
        return [
            ('average_q', self.average_q),
            ('average_loss', self.average_loss),
            ('n_updates', self.optimizer.t),
        ]
