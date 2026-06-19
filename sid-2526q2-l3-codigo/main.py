import os
import time
from tqdm import tqdm
from algorithms import JALGT, IQL
from solution_concepts import (
    MinimaxSolutionConcept,
    ParetoSolutionConcept,
    NashSolutionConcept,
    WelfareSolutionConcept
)
from game_model import GameModel
import numpy as np
from gymnasium import Wrapper
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
from utils import draw_history
import pandas as pd

# Parametros
MAP_SIZES = [4, 6, 10]
DENSITIES = [0.1, 0.3]
EPSILON_CONFIG = [(1.0, 0.1), (0.5, 0.01)]
GAMMA = [0.995, 0.95]
BASE_EPOCHS = 200 

POLYNOMIAL_DECAY = False
POLYNOMIAL_DECAY_DEGREE = 2
TRUNCATED_PENALTY = False
TRUNCATED_PENALIZED_REWARD = 0.2
USE_UNSEEN_MAPS = False
UNSEEN_SEED_OFFSET = 10000

EXPERIMENTS = [
    ("IQL", IQL, None),
    ("JAL-GT_Pareto", JALGT, ParetoSolutionConcept),
    ("JAL-GT_Nash", JALGT, NashSolutionConcept),
    ("JAL-GT_Minimax", JALGT, MinimaxSolutionConcept),
    ("JAL-GT_Welfare", JALGT, WelfareSolutionConcept)
]

def obs_to_state(obs):
    # Esta representación base asume observaciones de radio 1 (matrices 3x3).
    # Está pensada para el alcance obligatorio de la práctica; podéis cambiarla
    # si queréis experimentar con estados más complejos.
    matrix_obstacles = obs[0]
    matrix_agents = obs[1]
    matrix_target = obs[2]

    # Representación del objetivo:
    #  Ocupa 2 bits
    #  0 si el objetivo está arriba, diagonal arriba-izquierda o diagonal arriba-derecha
    #  1 si el objetivo está abajo, diagonal abajo-izquierda o diagonal abajo-derecha
    #  2 si el objetivo está a la izquierda (no en diagonal)
    #  3 si el objetivo está a la derecha (no en diagonal)
    target = np.max(matrix_target[2]) * 1 + \
             matrix_target[1][0] * 2 + matrix_target[1][2] * 3

    # Representación de los obstáculos:
    #  Shift de 2^6, ocupando 4 bits
    #  2^9 si hay un obstáculo arriba (no diagonal)
    #  2^8 si hay un obstáculo a la izquierda (no diagonal)
    #  2^7 si hay un obstáculo a la derecha (no diagonal)
    #  2^6 si hay un obstáculo abajo (no diagonal)
    obstacles = matrix_obstacles[0][1] * 2 ** 9 + \
                matrix_obstacles[1][0] * 2 ** 8 + \
                matrix_obstacles[1][2] * 2 ** 7 + \
                matrix_obstacles[2][1] * 2 ** 6

    # Representación de los otros agentes:
    #  Shift de 2^2, ocupando 4 bits
    #  2^5 si hay un agente arriba (no diagonal)
    #  2^4 si hay un agente a la izquierda (no diagonal)
    #  2^3 si hay un agente a la derecha (no diagonal)
    #  2^2 si hay un agente abajo (no diagonal)
    agents = matrix_agents[0][1] * 2 ** 5 + \
             matrix_agents[1][0] * 2 ** 4 + \
             matrix_agents[1][2] * 2 ** 3 + \
             matrix_agents[2][1] * 2 ** 2

    return int(obstacles + agents + target)


class RewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, joint_action):
        observations, rewards, terminated, truncated, infos = self.env.step(joint_action)
        for i in range(len(joint_action)):
            if TRUNCATED_PENALTY and truncated[i]:
                rewards[i] -= TRUNCATED_PENALIZED_REWARD
            elif not terminated[i] and rewards[i] == 0:
                rewards[i] -= 0.01
        return observations, rewards, terminated, truncated, infos


def create_env(config, seed=42):
    grid_config = GridConfig(num_agents=config["num_agents"],
                             size=config["size"],
                             density=config["obstacle_density"],
                             seed=seed,
                             max_episode_steps=config["episode_length"],
                             obs_radius=1,
                             on_target="finish",
                             render_mode=None)
    animation_config = AnimationConfig(directory=config["renders"],  # Dónde se guardarán las imágenes
                                       static=False,
                                       show_agents=True,
                                       egocentric_idx=None,  # Punto de vista
                                       save_every_idx_episode=config["save_every"],  # Guardar cada save_every episodios
                                       show_border=True,
                                       show_lines=True)
    env = pogema_v0(grid_config)
    env = AnimationMonitor(env, animation_config=animation_config)
    return RewardWrapper(env)  # Añadimos nuestra función de recompensa


def build_algorithms(config, game):
    algorithm_cls = config["algorithm_cls"]
    solution_concept_cls = config.get("solution_concept")
    algorithm_kwargs = dict(config.get("algorithm_kwargs", {}))

    algorithms = []
    for agent_id in range(game.num_agents):
        kwargs = dict(algorithm_kwargs)
        kwargs.setdefault("epsilon", config["epsilon_max"])
        kwargs.setdefault("alpha", config["learning_rate"])
        kwargs.setdefault("seed", agent_id)
        if solution_concept_cls is not None:
            kwargs.setdefault("solution_concept", solution_concept_cls())
        algorithms.append(algorithm_cls(agent_id, game, **kwargs))
    return algorithms


def compute_epsilon(config, global_episode):
    # Usamos un decay global por episodio para que el estudio de epsilon sea
    # fácil de interpretar y comparable entre algoritmos.
    total_episodes = config["epochs"] * config["episodes_per_epoch"]
    progress = global_episode / max(total_episodes - 1, 1)
    if POLYNOMIAL_DECAY:
        # funcion polinomial
        return config["epsilon_min"] + (config["epsilon_max"] - config["epsilon_min"]) * (1 - progress) ** POLYNOMIAL_DECAY_DEGREE
    else:
        return config["epsilon_max"] - progress * (config["epsilon_max"] - config["epsilon_min"])


def train_episode(env, algorithms, game, epsilon):
    for algorithm in algorithms:
        algorithm.set_epsilon(epsilon)

    observations, infos = env.reset()
    terminated = [False] * game.num_agents
    truncated = [False] * game.num_agents
    total_rewards = [0] * game.num_agents
    td_errors = []
    states = [obs_to_state(observations[i]) for i in range(game.num_agents)]

    while not all(terminated) and not all(truncated):
        actions = tuple(algorithms[i].select_action(states[i]) for i in range(game.num_agents))
        observations, rewards, terminated, truncated, infos = env.step(actions)
        next_states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
        for i in range(game.num_agents):
            algorithms[i].learn(actions, rewards, states[i], next_states[i])
            td_errors.append(algorithms[i].metrics["td_error"][-1])
        total_rewards = [total_rewards[i] + rewards[i] for i in range(game.num_agents)]
        states = next_states

    return total_rewards, td_errors


def evaluate_episode(env, algorithms, game):
    observations, infos = env.reset()
    terminated = [False] * game.num_agents
    truncated = [False] * game.num_agents
    total_rewards = [0] * game.num_agents
    states = [obs_to_state(observations[i]) for i in range(game.num_agents)]

    while not all(terminated) and not all(truncated):
        actions = tuple(algorithms[i].select_action(states[i], train=False) for i in range(game.num_agents))
        observations, rewards, terminated, truncated, infos = env.step(actions)
        total_rewards = [total_rewards[i] + rewards[i] for i in range(game.num_agents)]
        states = [obs_to_state(observations[i]) for i in range(game.num_agents)]

    return total_rewards


def get_optimal_policies(algorithms, game):
    policies = {}
    for agent_id, algorithm in enumerate(algorithms):
        agent_policy = []
        for state in range(game.num_states):
            explanation = algorithm.explain(state)
            if isinstance(algorithm, IQL): # IQL
                state_policy = int(explanation["policy_action"])
            else: # JAL-GT
                state_policy = [float(probability) for probability in explanation["policy"]]
            agent_policy.append(state_policy)
        policies[f"Optimal_Policy_Agent_{agent_id}"] = agent_policy
    return policies


if __name__ == '__main__':
    
    # Parametros
    map_sizes = MAP_SIZES
    densities = DENSITIES
    epsilon_config = EPSILON_CONFIG
    gamma_values = GAMMA
    base_epochs = BASE_EPOCHS
    experiments = EXPERIMENTS

    all_results = []

    for size in map_sizes:
        for density in densities:
            for (epsilon_max, epsilon_min) in epsilon_config:
                for gamma in gamma_values:
                    for exp_name, alg_cls, sol_cls in experiments:
                        print(f"\n--- Ejecutando: {exp_name} | Mapa: {size}x{size} | Densidad: {density} | Epsilon_Max: {epsilon_max} | Epsilon_Min: {epsilon_min} | Gamma: {gamma} ---")

                        exp_config = {
                            "num_agents": 2, # Número de agentes
                            "size": size,
                            "maps": 10, # Número de mapas a entrenar y evaluar (se repiten si episodios > mapas)
                            "num_states": 16 * 16 * 4, # Obstacle representation x Agent representation x Target representation
                            "epochs": base_epochs, # Cada epoch es un entrenamiento de un número de episodios y una evaluación
                            "episodes_per_epoch": 10,  # Número mínimo de episodios por epoch de entrenamiento
                            "episode_length": 16,  # Número máximo de pasos por episodio, se trunca si se excede
                            "obstacle_density": density, # Probabilidad de tener un obstáculo en el mapa
                            "save_every": None,  # Frecuencia con que se guarda el SVG con la animación de la ejecución
                            "learning_rate": 0.01,  # alpha
                            "epsilon_max": epsilon_max,  # epsilon inicial del entrenamiento
                            "epsilon_min": epsilon_min,  # cota mínima de epsilon
                            "renders": f"renders/{exp_name}_{size}x{size}_d{density}{'_unseen' if USE_UNSEEN_MAPS else ''}/", # directorio donde generar las animaciones
                            "algorithm_cls": alg_cls, # Aquí podéis conectar IQL u otro algoritmo.
                            "algorithm_kwargs": {"gamma": gamma},
                            "solution_concept": sol_cls
                        }

                        # Creamos el directorio de renders si no existe ya.
                        os.makedirs(exp_config["renders"], exist_ok=True)

                        # Modelo de juego y algoritmos (uno para cada agente)
                        game = GameModel(num_agents=exp_config["num_agents"], num_states=exp_config["num_states"], num_actions=5)
                        # STAY, UP, DOWN, LEFT, RIGHT
                        algorithms = build_algorithms(exp_config, game)

                        # Variables para almacenar métricas
                        reward_per_epoch = []
                        individual_reward_per_epoch = [[] for _ in range(game.num_agents)]
                        td_error_per_epoch = []

                        pbar = tqdm(range(exp_config["epochs"])) # Barra de progreso

                        for epoch in pbar:
                            epoch_start_time = time.perf_counter()
                            all_eval_rewards = []
                            all_td_errors = []
                            agent_rewards = [[] for _ in range(game.num_agents)]

                            # ENTRENAMIENTO
                            for ep in range(exp_config["episodes_per_epoch"]):
                                global_episode = epoch * exp_config["episodes_per_epoch"] + ep
                                epsilon = compute_epsilon(exp_config, global_episode)
                                env = create_env(config=exp_config, seed=ep % exp_config["maps"])
                                _, td_errors = train_episode(env, algorithms, game, epsilon)
                                all_td_errors.extend(td_errors)

                            avg_td_error = sum(np.abs(all_td_errors)) / len(all_td_errors) if all_td_errors else 0

                            # EVALUACIÓN
                            for ep in range(exp_config["maps"]):
                                evaluation_seed = ep + UNSEEN_SEED_OFFSET if USE_UNSEEN_MAPS else ep
                                env = create_env(config=exp_config, seed=evaluation_seed)
                                total_rewards = evaluate_episode(env, algorithms, game)

                                # solo renderiza ultimo epoch
                                if epoch == exp_config["epochs"] - 1:
                                    for agent_i in range(exp_config["num_agents"]):
                                        env.save_animation(f"{exp_config['renders']}/map{evaluation_seed}-agent{agent_i}.svg",
                                                           AnimationConfig(egocentric_idx=agent_i, show_border=True, show_lines=True))

                                all_eval_rewards.append(sum(total_rewards))
                                for i, reward in enumerate(total_rewards):
                                    agent_rewards[i].append(reward)

                            epoch_time = time.perf_counter() - epoch_start_time
                            episode_count = exp_config["episodes_per_epoch"] + exp_config["maps"]

                            avg_reward = sum(all_eval_rewards) / len(all_eval_rewards)
                            avg_agent_rewards = [
                                float(np.mean(rewards)) for rewards in agent_rewards
                            ]
                            reward_per_epoch.append(avg_reward)
                            td_error_per_epoch.append(avg_td_error)
                            for agent_id in range(game.num_agents):
                                individual_reward_per_epoch[agent_id].append(avg_agent_rewards[agent_id])

                            if epoch == exp_config["epochs"] - 1: # solo se guarda ultimo epoch
                                optimal_policies = get_optimal_policies(algorithms, game)
                            else:
                                optimal_policies = {
                                    f"Optimal_Policy_Agent_{agent_id}": None
                                    for agent_id in range(game.num_agents)
                                }
                            pbar.set_description(f"R: {avg_reward:.2f} | TD: {avg_td_error:.4f}")

                            degree = POLYNOMIAL_DECAY_DEGREE if POLYNOMIAL_DECAY else None

                            all_results.append({
                                "Algorithm": exp_name,
                                "Size": size,
                                "Density": density,
                                "Epsilon_Max": epsilon_max,
                                "Epsilon_Min": epsilon_min,
                                "Gamma": gamma,
                                "Epoch": epoch,
                                "Epoch_Time_Seconds": epoch_time,
                                "Episodes": episode_count,
                                "Time_Per_Episode_Seconds": epoch_time / episode_count,
                                "Avg_Agent_0_Reward": avg_agent_rewards[0],
                                "Avg_Agent_1_Reward": avg_agent_rewards[1],
                                "Avg_Collective_Reward": avg_reward,
                                "Avg_TD_Error": avg_td_error,
                                "Alternative_Polinomial_Decay": POLYNOMIAL_DECAY,
                                "Polinomial_Decay_Degree": degree,
                                "Truncated_penalty": TRUNCATED_PENALTY,
                                "Evaluation_Map_Type": "Unseen" if USE_UNSEEN_MAPS else "Seen",
                                **optimal_policies
                            })

                        graph_dir = os.path.join(
                            "graficas_memoria",
                            f"{exp_name}_{size}x{size}_d{density}_eps{epsilon_max}-{epsilon_min}_g{gamma}"
                            f"{'_unseen' if USE_UNSEEN_MAPS else ''}"
                        )
                        draw_history(
                            reward_per_epoch,
                            "Collective Reward",
                            os.path.join(graph_dir, "collective_reward.png")
                        )
                        draw_history(
                            {
                                f"Agent {agent_id}": history
                                for agent_id, history in enumerate(individual_reward_per_epoch)
                            },
                            "Individual Reward",
                            os.path.join(graph_dir, "individual_reward.png")
                        )
                        draw_history(
                            td_error_per_epoch,
                            "TD Error",
                            os.path.join(graph_dir, "td_error.png")
                        )

    df = pd.DataFrame(all_results)
    df.to_csv("resultados_experimentos.csv", index=False)
    print("\nExperimentos terminados y resultados guardados en 'resultados_experimentos.csv'.")
