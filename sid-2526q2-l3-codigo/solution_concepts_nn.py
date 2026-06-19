import itertools
import numpy as np

from solution_concepts import SolutionConcept, softmax
from jalgtnn import one_hot


class ParetoSolutionConcept(SolutionConcept):
    def is_dominated_cached(
        self,
        joint_action,
        other_joint_action,
        game,
        q_cache,
    ):
        strictly_better_for_at_least_one = False

        joint_action_index = game.action_space_index[joint_action]
        other_joint_action_index = game.action_space_index[other_joint_action]
        for agent_id in range(game.num_agents):

            q_state_a_i = q_cache[agent_id][joint_action_index]
            q_state_a_j = q_cache[agent_id][other_joint_action_index]

            if q_state_a_j < q_state_a_i:
                return False

            if q_state_a_j > q_state_a_i:
                strictly_better_for_at_least_one = True

        return strictly_better_for_at_least_one

    def find_pareto_efficient_solutions(self, state, game, q_models):

        joint_actions = game.action_space

        state_oh = one_hot(state, game.num_states)

        q_cache = [q_models[i].forward(state_oh)[0] for i in range(game.num_agents)]

        pareto_solutions = []

        for joint_action in joint_actions:

            dominated = False

            for other_joint_action in joint_actions:

                if other_joint_action == joint_action:
                    continue

                if self.is_dominated_cached(
                    joint_action,
                    other_joint_action,
                    game,
                    q_cache,
                ):
                    dominated = True
                    break

            if not dominated:
                pareto_solutions.append(joint_action)

        return pareto_solutions

    def solution_policy(self, agent_id, state, game, q_models):
        pareto_solutions = self.find_pareto_efficient_solutions(state, game, q_models)
        if len(pareto_solutions) > 0:
            action_counts = np.zeros(game.num_actions)
            for pareto_solution in pareto_solutions:
                action_counts[pareto_solution[agent_id]] += 1
            probs = action_counts / len(pareto_solutions)
            return probs
        else:
            uniform_distribution = np.ones(game.num_actions)
            return uniform_distribution / np.sum(uniform_distribution)


class MinimaxSolutionConcept(SolutionConcept):
    def opponent_max_values(self, agent_id, scalar_state, game, q_models):
        action_scores = []
        q_opponent = q_models[1 - agent_id]
        state = one_hot(scalar_state, game.num_states)
        for action in range(game.num_actions):
            max_opponent_payoff = -999999999999
            for opponent_action in range(game.num_actions):
                if agent_id == 0:  # Suponemos sólo dos agentes
                    joint_action = (action, opponent_action)
                else:
                    joint_action = (opponent_action, action)
                joint_action_index = game.action_space_index[joint_action]
                score = q_opponent.forward(state)[0][joint_action_index]
                if score > max_opponent_payoff:
                    max_opponent_payoff = score
            action_scores.append(max_opponent_payoff)
        return np.array(action_scores)

    def solution_policy(self, agent_id, state, game, q_models):
        vals = np.array(self.opponent_max_values(agent_id, state, game, q_models))
        return softmax(-vals)


class NashSolutionConcept(SolutionConcept):
    def generate_others_actions(self, fixed_agent_id, num_agents, num_actions):
        # Lista con las estrategias que pueden seguir los demás agentes
        strategies_minus_me = [
            range(num_actions) if i != fixed_agent_id else [None]
            for i in range(num_agents)
        ]
        # itertools.product nos da el producto cartesiano
        return list(itertools.product(*strategies_minus_me))

    def calculate_best_responses(self, agent_id, scalar_state, game, q_models):
        state = one_hot(scalar_state, game.num_states)
        others_joint_actions = self.generate_others_actions(
            agent_id, game.num_agents, game.num_actions
        )
        best_responses = set()
        q_state = q_models[agent_id].forward(state)[0]

        for joint_action in others_joint_actions:
            payoffs = []
            for action in range(game.num_actions):
                joint_action_copy = list(joint_action)
                joint_action_copy[agent_id] = action
                full_joint_action = tuple(joint_action_copy)
                joint_action_index = game.action_space_index[full_joint_action]
                payoff = q_state[joint_action_index]
                payoffs.append((full_joint_action, payoff))

            max_payoff = max(payoff for _, payoff in payoffs)
            for full_joint_action, payoff in payoffs:
                if payoff == max_payoff:
                    best_responses.add(full_joint_action)

        return best_responses

    def find_nash_equilibria(self, state, game, q_models):
        best_responses = [
            self.calculate_best_responses(i, state, game, q_models)
            for i in range(game.num_agents)
        ]
        return list(set.intersection(*best_responses))

    def solution_policy(self, agent_id, state, game, q_models):
        nash_equilibria = self.find_nash_equilibria(state, game, q_models)
        if len(nash_equilibria) > 0:
            equilibrium = nash_equilibria[0]
            probs = np.zeros(game.num_actions)
            probs[equilibrium[agent_id]] = 1
            return probs
        else:
            uniform_distribution = [1] * game.num_actions
            return uniform_distribution / np.sum(uniform_distribution)


class WelfareSolutionConcept(SolutionConcept):
    def find_welfare_maximizing_solutions(self, state, game, q_models):
        joint_actions = game.action_space
        state_oh = one_hot(state, game.num_states)

        q_cache = [q_models[i].forward(state_oh)[0] for i in range(game.num_agents)]
        welfare_values = []

        for joint_action in joint_actions:

            joint_action_index = game.action_space_index[joint_action]

            welfare = 0

            for agent_id in range(game.num_agents):
                welfare += q_cache[agent_id][joint_action_index]

            welfare_values.append((welfare, joint_action))

        max_welfare = max(welfare_values, key=lambda x: x[0])[0]

        welfare_maximizing_solutions = [
            action for welfare, action in welfare_values if welfare == max_welfare
        ]

        return welfare_maximizing_solutions

    def solution_policy(self, agent_id, state, game, q_models):

        welfare_solutions = self.find_welfare_maximizing_solutions(
            state,
            game,
            q_models,
        )

        num_solutions = len(welfare_solutions)

        if num_solutions > 0:

            probs = np.zeros(game.num_actions)

            for solution in welfare_solutions:
                probs[solution[agent_id]] += 1 / num_solutions

            return probs

        else:

            uniform_distribution = np.ones(game.num_actions)

            return uniform_distribution / np.sum(uniform_distribution)
