#-*-coding:utf-8-*-
#遗传算法优化器
import numpy as np
import random
from typing import List, Dict, Callable, Tuple
import matplotlib.pyplot as plt


class GeneticAlgorithmOptimizer:
    """
    通用的遗传算法优化器，用于寻找函数最优解
    """

    def __init__(self, objective_function: Callable,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_count: int = 2,
                 max_generations: int = 50):
        """
        初始化遗传算法优化器

        Parameters:
        objective_function: 目标函数，接受一个参数向量，返回一个标量值（需要最小化）
        bounds: 每个参数的边界列表，格式为 [(min1, max1), (min2, max2), ...]
        population_size: 种群大小
        mutation_rate: 变异率
        crossover_rate: 交叉率
        elitism_count: # 精英保留数量（直接遗传到下一代）
        max_generations: 最大迭代代数
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.max_generations = max_generations
        self.num_variables = len(bounds)

        # 记录优化过程
        self.best_fitness_history = []      # 每代最佳适应度
        self.avg_fitness_history = []       # 每代平均适应度
        self.worst_fitness_history = []     # 每代最差适应度
        self.best_individual_history = []   #每代最优参数值

    def initialize_population(self) -> np.ndarray:
        """
        初始化种群：在参数边界内随机生成个体（每个个体是一个参数向量）

        Returns:
        初始种群，形状为 (population_size, num_variables)
        """
        population = np.zeros((self.population_size, self.num_variables))
        for i in range(self.num_variables):
            min_val, max_val = self.bounds[i]
            population[:, i] = np.random.uniform(min_val, max_val, self.population_size)
        return population

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        评估种群中每个个体的适应度：对每个个体调用目标函数（计算能耗）

        Parameters:
        population: 种群矩阵

        Returns:
        适应度数组
        """
        print("Population shape:", population.shape)  # 添加这行
        # 添加边界检查
        if population.shape[0] == 0:
            return np.array([])
        fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitness[i] = self.objective_function(population[i, :])
        return fitness

    def select_parents(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        # 选择父代（锦标赛选择法）：随机选3个个体，保留适应度最好的（能耗最小的）

        Parameters:
        population: 种群矩阵
        fitness: 适应度数组

        Returns:
        选择的父代个体
        """
        # 锦标赛选择
        selected_indices = []
        tournament_size = 3

        # 修复：选择与种群大小相同数量的父代，而不是减去精英数量
        for _ in range(self.population_size):  # 原为 self.population_size - self.elitism_count
            # 随机选择 tournament_size 个个体
            tournament_indices = np.random.choice(range(self.population_size), tournament_size, replace=False)
            # 选择适应度最好的个体（最小化问题，适应度越小越好）
            winner_index = tournament_indices[np.argmin(fitness[tournament_indices])]
            selected_indices.append(winner_index)

        return population[selected_indices, :]

    def crossover(self, parents: np.ndarray) -> np.ndarray:
        """
        交叉操作（模拟二进制交叉）：父代基因重组生成子代

        Parameters:
        parents: 父代个体

        Returns:
        子代个体
        """
        offspring = np.zeros_like(parents)

        for i in range(0, parents.shape[0], 2):
            if i + 1 < parents.shape[0] and random.random() < self.crossover_rate:
                # 对每对父代进行交叉
                parent1 = parents[i, :]
                parent2 = parents[i + 1, :]

                # 模拟二进制交叉
                eta = 20  # 分布指数
                u = random.random()

                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                # 生成子代
                offspring[i, :] = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
                offspring[i + 1, :] = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

                # 确保子代在边界内
                for j in range(self.num_variables):
                    min_val, max_val = self.bounds[j]
                    offspring[i, j] = np.clip(offspring[i, j], min_val, max_val)
                    offspring[i + 1, j] = np.clip(offspring[i + 1, j], min_val, max_val)
            else:
                # 不进行交叉，直接复制父代
                if i < parents.shape[0]:
                    offspring[i, :] = parents[i, :]
                if i + 1 < parents.shape[0]:
                    offspring[i + 1, :] = parents[i + 1, :]

        return offspring

    def mutate(self, offspring: np.ndarray) -> np.ndarray:
        """
        变异操作（多项式变异）：随机改变子代基因，增加多样性

        Parameters:
        offspring: 子代个体

        Returns:
        变异后的子代个体
        """
        for i in range(offspring.shape[0]):
            for j in range(self.num_variables):
                if random.random() < self.mutation_rate:
                    # 多项式变异
                    min_val, max_val = self.bounds[j]
                    delta = max_val - min_val

                    eta = 20  # 分布指数
                    u = random.random()

                    if u <= 0.5:
                        delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

                    # 应用变异
                    offspring[i, j] += delta_q * delta

                    # 确保变异后在边界内
                    offspring[i, j] = np.clip(offspring[i, j], min_val, max_val)

        return offspring

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        执行遗传算法优化：迭代执行选择→交叉→变异→精英保留

        Returns:
        最优解和最优适应度
        """
        # 初始化种群
        population = self.initialize_population()

        # 评估初始种群
        fitness = self.evaluate_population(population)

        # 记录初始最优解
        best_fitness = np.min(fitness)
        best_individual = population[np.argmin(fitness), :]

        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(np.mean(fitness))
        self.worst_fitness_history.append(np.max(fitness))
        self.best_individual_history.append(best_individual.copy())

        print(f"初始最佳适应度: {best_fitness:.4f}")

        # 开始进化
        for generation in range(self.max_generations):
            # 选择
            parents = self.select_parents(population, fitness)

            # 交叉
            offspring = self.crossover(parents)

            # 变异
            offspring = self.mutate(offspring)

            # 评估子代
            offspring_fitness = self.evaluate_population(offspring)

            # 精英保留：将当前种群中最好的几个个体替换子代中最差的几个个体
            if self.elitism_count > 0:
                # 找到当前种群中最好的个体
                elite_indices = np.argsort(fitness)[:self.elitism_count]
                elite_individuals = population[elite_indices, :]
                elite_fitness = fitness[elite_indices]

                # 找到子代中最差的个体
                worst_indices = np.argsort(offspring_fitness)[-self.elitism_count:]

                # 用精英个体替换最差个体
                offspring[worst_indices, :] = elite_individuals
                offspring_fitness[worst_indices] = elite_fitness

            # 更新种群
            population = offspring
            fitness = offspring_fitness

            # 更新最佳解
            current_best_fitness = np.min(fitness)
            current_best_individual = population[np.argmin(fitness), :]

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual

            # 记录历史
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitness))
            self.worst_fitness_history.append(np.max(fitness))
            self.best_individual_history.append(best_individual.copy())

            print(f"代 {generation+1}/{self.max_generations}, 最佳适应度: {best_fitness:.4f}")

        return best_individual, best_fitness

    def plot_optimization_history(self, save_path: str = None):
        """
        绘制优化历史

        Parameters:
        save_path: 图片保存路径
        """
        # 设置中文字体，解决乱码问题
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        # 解决负号显示问题（可选）
        plt.rcParams['axes.unicode_minus'] = False
        generations = range(len(self.best_fitness_history))

        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.best_fitness_history, 'b-', label='最佳适应度')
        plt.plot(generations, self.avg_fitness_history, 'g-', label='平均适应度')
        plt.plot(generations, self.worst_fitness_history, 'r-', label='最差适应度')

        plt.xlabel('代数')
        plt.ylabel('适应度')
        plt.title('遗传算法优化历史')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"优化历史图表已保存到 {save_path}")

        plt.show()