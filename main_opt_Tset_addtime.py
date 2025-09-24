# -*-coding:utf-8-*-
import os
import matplotlib.pyplot as plt
import numpy as np
from genetic_algorithm import GeneticAlgorithmOptimizer
from cooling_system_simulator import CoolingSystemSimulator
import warnings
import sys
from contextlib import redirect_stderr
import time  # 新增：导入时间模块

# 禁用警告以保持输出整洁
warnings.filterwarnings("ignore")


def main():
    """主函数：集成遗传算法和制冷系统仿真器，仅优化Tset参数"""
    # 1. 配置文件路径（保持不变）
    FMU_PATH = r"C:\Users\xxp\Documents\Dymola\CET-zhuzhouyidong\simulation_changeinput\CoolingStation_0changeinput_System_Coolingstation.fmu"
    INPUT_CSV_PATH = ".\数据文件\Input\input_data.csv"  # 其他参数从该CSV文件导入

    # 2. 初始化仿真器（保持不变）
    simulator = CoolingSystemSimulator(FMU_PATH)

    # 3. 查看可用的输入输出变量（可选，保持不变）
    print("输入变量:")
    for name, type_, causality in simulator.get_input_variables():
        print(f"  {name} ({type_}, {causality})")

    print("\n输出变量:")
    for name, type_, causality in simulator.get_output_variables():
        print(f"  {name} ({type_}, {causality})")

    # 4. 定义目标函数（保持不变，仅添加时间记录）
    def objective_function(x):
        tset_value = x[0]
        print(f"尝试 Tset={tset_value-273.15:.2f}°C")

        # 记录单个体仿真开始时间
        sim_start = time.time()
        with open(os.devnull, 'w') as f, redirect_stderr(f):
            try:
                success = simulator.simulate(
                    start_time=start_time,
                    stop_time=stop_time,
                    step_size=step_size,
                    input_values={'Tset': tset_value},
                    input_csv_path=INPUT_CSV_PATH
                )
            except Exception as e:
                print(f"仿真出错: {e}")
                return 1e10

        # 计算单个体仿真耗时
        sim_time = time.time() - sim_start
        print(f"单个体仿真耗时: {sim_time:.2f}秒")

        if success:
            energy = simulator.total_energy / 3.6e6
            print(f"总能耗: {energy:.2f} kWh")
            return energy
        else:
            print("仿真失败，返回大能耗值")
            return 1e10

    # 5. 仿真与优化参数配置（保持不变）
    start_time = 0.0
    stop_time = 7200.0
    step_size = 5.0

    optimization_vars = [
        {
            "name": "Tset",
            "bounds": (273.15 + 16, 273.15 + 20),
            "unit": "K",
            "scale": 1.0,
            "scale_unit": "°C",
            "offset": 273.15
        }
    ]
    bounds = [var["bounds"] for var in optimization_vars]

    # 6. 运行遗传算法优化（添加总时间和迭代时间记录）
    print("\n开始遗传算法优化...")
    ga_optimizer = GeneticAlgorithmOptimizer(
        objective_function=objective_function,
        bounds=bounds,
        population_size=10,
        max_generations=5,
        mutation_rate=0.15,
        crossover_rate=0.8
    )

    # 记录优化总开始时间
    total_start = time.time()

    # 重写优化循环以记录每代耗时
    def timed_optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate_population(population)

        best_fitness = np.min(fitness)
        best_individual = population[np.argmin(fitness), :]

        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(np.mean(fitness))
        self.worst_fitness_history.append(np.max(fitness))
        self.best_individual_history.append(best_individual.copy())

        print(f"初始最佳适应度: {best_fitness:.4f}")

        # 记录每代耗时
        generation_times = []
        for generation in range(self.max_generations):
            gen_start = time.time()  # 每代开始时间

            parents = self.select_parents(population, fitness)
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)
            offspring_fitness = self.evaluate_population(offspring)

            if self.elitism_count > 0:
                elite_indices = np.argsort(fitness)[:self.elitism_count]
                elite_individuals = population[elite_indices, :]
                elite_fitness = fitness[elite_indices]
                worst_indices = np.argsort(offspring_fitness)[-self.elitism_count:]
                offspring[worst_indices, :] = elite_individuals
                offspring_fitness[worst_indices] = elite_fitness

            population = offspring
            fitness = offspring_fitness

            current_best_fitness = np.min(fitness)
            current_best_individual = population[np.argmin(fitness), :]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual

            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitness))
            self.worst_fitness_history.append(np.max(fitness))
            self.best_individual_history.append(best_individual.copy())

            # 计算并记录每代耗时
            gen_time = time.time() - gen_start
            generation_times.append(gen_time)
            print(f"代 {generation+1}/{self.max_generations}, 最佳适应度: {best_fitness:.4f}, 耗时: {gen_time:.2f}秒")

        return best_individual, best_fitness, generation_times

    # 执行带时间记录的优化
    best_solution, best_fitness, generation_times = timed_optimize(ga_optimizer)
    total_time = time.time() - total_start  # 计算总耗时

    # 7. 绘制优化历史（保持不变）
    ga_optimizer.plot_optimization_history('.\数据文件\Output\ga_optimization_history.png')

    # 8. 输出优化结果（新增时间统计）
    print(f"\n优化完成!")
    # 输出每代耗时
    for i, t in enumerate(generation_times, 1):
        print(f"第{i}代耗时: {t:.2f}秒")
    # 输出总耗时
    print(f"优化总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")

    # 输出最优参数（保持不变）
    for i, var in enumerate(optimization_vars):
        value = best_solution[i]
        display_value = value - var["offset"]
        print(f"最优 {var['name']}: {display_value:.2f} {var['scale_unit']} ({value:.2f} {var['unit']})")
    print(f"最小能耗: {best_fitness:.2f} kWh")

    # 9. 最终精确仿真（保持不变）
    # ...（省略部分代码）


if __name__ == "__main__":
    main()