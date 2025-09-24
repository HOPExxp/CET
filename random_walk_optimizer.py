#-*-coding:utf-8-*-
# -*-coding:utf-8-*-
# 随机走步优化器
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List


class RandomWalkOptimizer:
    """
    随机走步优化器，用于寻找函数最优解（最小化问题）
    """

    def __init__(self,
                 objective_function: Callable,
                 bounds: List[Tuple[float, float]],
                 initial_step_size: float = 0.1,
                 step_decay: float = 0.95,
                 max_iterations: int = 100,
                 num_walkers: int = 5):
        """
        初始化随机走步优化器

        Parameters:
        objective_function: 目标函数，接受参数向量返回标量值（需最小化）
        bounds: 参数边界列表，格式为[(min1, max1), (min2, max2), ...]
        initial_step_size: 初始步长比例（相对于参数范围）
        step_decay: 步长衰减系数（每轮迭代后乘以该系数）
        max_iterations: 最大迭代次数
        num_walkers: 并行走步数量，增加搜索多样性
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_variables = len(bounds)

        # 计算参数范围用于步长缩放
        self.param_ranges = np.array([max_val - min_val for min_val, max_val in bounds])

        self.initial_step_size = initial_step_size
        self.step_decay = step_decay
        self.max_iterations = max_iterations
        self.num_walkers = num_walkers

        # 记录优化历史
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_solution_history = []

        # 初始化最优解
        self.best_solution = None
        self.best_fitness = float('inf')

    def _initialize_walkers(self) -> np.ndarray:
        """初始化随机走步起点"""
        walkers = np.zeros((self.num_walkers, self.num_variables))
        for i in range(self.num_variables):
            min_val, max_val = self.bounds[i]
            walkers[:, i] = np.random.uniform(min_val, max_val, self.num_walkers)
        return walkers

    def _take_step(self, current_positions, step_size) -> np.ndarray:
        """生成新的走步位置"""
        new_positions = []
        for pos in current_positions:
            # 生成随机步长（正态分布）
            step = np.random.normal(0, step_size, self.num_variables) * self.param_ranges
            new_pos = pos + step

            # 确保在边界内
            for i in range(self.num_variables):
                min_val, max_val = self.bounds[i]
                new_pos[i] = np.clip(new_pos[i], min_val, max_val)

            new_positions.append(new_pos)
        return np.array(new_positions)

    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行随机走步优化"""
        # 初始化走步者
        current_positions = self._initialize_walkers()
        current_fitness = np.array([self.objective_function(pos) for pos in current_positions])

        # 初始化最优解
        self.best_solution = current_positions[np.argmin(current_fitness)]
        self.best_fitness = np.min(current_fitness)

        # 记录初始状态
        self.best_fitness_history.append(self.best_fitness)
        self.avg_fitness_history.append(np.mean(current_fitness))
        self.best_solution_history.append(self.best_solution.copy())

        print(f"初始最佳适应度: {self.best_fitness:.4f}")

        # 当前步长
        current_step_size = self.initial_step_size

        # 迭代优化
        for iteration in range(self.max_iterations):
            # 生成新位置
            new_positions = self._take_step(current_positions, current_step_size)
            new_fitness = np.array([self.objective_function(pos) for pos in new_positions])

            # 接受更优解
            for i in range(self.num_walkers):
                if new_fitness[i] < current_fitness[i]:
                    current_positions[i] = new_positions[i]
                    current_fitness[i] = new_fitness[i]

            # 更新全局最优
            current_best_idx = np.argmin(current_fitness)
            if current_fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = current_fitness[current_best_idx]
                self.best_solution = current_positions[current_best_idx].copy()

            # 记录历史
            self.best_fitness_history.append(self.best_fitness)
            self.avg_fitness_history.append(np.mean(current_fitness))
            self.best_solution_history.append(self.best_solution.copy())

            # 衰减步长
            current_step_size *= self.step_decay

            print(
                f"迭代 {iteration+1}/{self.max_iterations}, 最佳适应度: {self.best_fitness:.4f}, 当前步长: {current_step_size:.4f}")

        return self.best_solution, self.best_fitness

    def plot_optimization_history(self, save_path: str = None):
        """绘制优化历史曲线"""
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams['axes.unicode_minus'] = False

        iterations = range(len(self.best_fitness_history))

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.best_fitness_history, 'b-', label='最佳适应度')
        plt.plot(iterations, self.avg_fitness_history, 'g-', label='平均适应度')

        plt.xlabel('迭代次数')
        plt.ylabel('适应度（能耗）')
        plt.title('随机走步优化历史')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"优化历史图表已保存到 {save_path}")

        plt.show()