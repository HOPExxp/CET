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

warnings.filterwarnings("ignore")


def main():
    """
    主函数：集成遗传算法和制冷系统仿真器，优化Tset和coolingLoad参数
    """
    # 1. 配置文件路径
    FMU_PATH = r"C:\Users\xxp\Documents\Dymola\CET-zhuzhouyidong\simulation_changeinput\CoolingStation_0changeinput_System_Coolingstation.fmu"
    INPUT_CSV_PATH = ".\数据文件\Input\input_data.csv"  # 其他参数从该CSV文件导入

    # 2. 初始化仿真器
    simulator = CoolingSystemSimulator(FMU_PATH)

    # 3. 查看可用的输入输出变量（可选，用于调试）
    print("输入变量:")
    for name, type_, causality in simulator.get_input_variables():
        print(f"  {name} ({type_}, {causality})")

    print("\n输出变量:")
    for name, type_, causality in simulator.get_output_variables():
        print(f"  {name} ({type_}, {causality})")

    # 4. 定义目标函数（优化Tset和coolingLoad）
    def objective_function(x):
        """
        目标函数：计算给定参数下的系统能耗

        Parameters:
        x: 参数向量，格式为 [Tset, coolingLoad]（Tset单位：开尔文，coolingLoad单位：W）

        Returns:
        系统总能耗（kWh），值越小越优
        """
        tset_value = x[0]
        cooling_load_value = x[1]

        # 转换为更易读的单位显示
        print(f"尝试 Tset={tset_value-273.15:.2f}°C, coolingLoad={cooling_load_value/1e6:.2f}MW")

        # 运行仿真（重定向stderr避免冗余输出）
        with open(os.devnull, 'w') as f, redirect_stderr(f):
            try:
                success = simulator.simulate(
                    start_time=start_time,
                    stop_time=stop_time,
                    step_size=step_size,
                    input_values={
                        'Tset': tset_value,
                        'coolingLoad': cooling_load_value
                    },  # 传递两个待优化参数
                    input_csv_path=INPUT_CSV_PATH  # 其他参数从CSV导入
                )
            except Exception as e:
                print(f"仿真出错: {e}")
                return 1e10  # 错误时返回大惩罚值

        if success:
            energy = simulator.total_energy / 3.6e6  # 转换为kWh
            print(f"总能耗: {energy:.2f} kWh")
            return energy
        else:
            print("仿真失败，返回大能耗值")
            return 1e10  # 失败时返回大惩罚值

    # 5. 仿真与优化参数配置
    start_time = 0.0
    stop_time = 7200.0  # 2小时仿真
    step_size = 5.0  # 优化阶段步长

    # 优化变量配置（增加coolingLoad参数）
    optimization_vars = [
        {
            "name": "Tset",
            "bounds": (273.15 + 16, 273.15 + 20),  # Tset范围：16-20°C（转换为开尔文）
            "unit": "K",
            "scale": 1.0,
            "scale_unit": "°C",
            "offset": 273.15  # 开尔文到摄氏度的转换偏移量
        },
        {
            "name": "coolingLoad",
            "bounds": (5e6, 10e6),  # 冷负荷范围：5-10MW（根据实际情况调整）
            "unit": "W",
            "scale": 1e-6,
            "scale_unit": "MW",
            "offset": 0  # 不需要偏移量
        }
    ]
    # 提取边界用于遗传算法
    bounds = [var["bounds"] for var in optimization_vars]

    # 6. 运行遗传算法优化（适当调整种群大小和迭代次数以提高优化效果）
    print("\n开始遗传算法优化...")
    ga_optimizer = GeneticAlgorithmOptimizer(
        objective_function=objective_function,
        bounds=bounds,
        population_size=10,  # 增加种群大小以覆盖更多可能解
        max_generations=5,  # 增加迭代次数
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

    # 7. 绘制优化历史
    ga_optimizer.plot_optimization_history('.\数据文件\Output\ga_optimization_history.png')

    # 8. 输出优化结果
    print(f"\n优化完成!")
    # 输出每代耗时
    for i, t in enumerate(generation_times, 1):
        print(f"第{i}代耗时: {t:.2f}秒")
    # 输出总耗时
    print(f"优化总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    for i, var in enumerate(optimization_vars):
        value = best_solution[i]
        # 转换为更易读的单位显示
        display_value = value * var["scale"] - var["offset"]
        print(f"最优 {var['name']}: {display_value:.2f} {var['scale_unit']} ({value:.2f} {var['unit']})")
    print(f"最小能耗: {best_fitness:.2f} kWh")

    # 9. 最终精确仿真（使用优化结果）
    # print("\n使用最优参数运行最终仿真...")
    # final_step_size = 5.0  # 更小步长提高精度
    #
    # with open(os.devnull, 'w') as f, redirect_stderr(f):
    #     try:
    #         optimal_success = simulator.simulate(
    #             start_time=start_time,
    #             stop_time=stop_time,
    #             step_size=final_step_size,  # 使用更小的步长
    #             input_values={
    #                 'Tset': best_solution[0],
    #                 'coolingLoad': best_solution[1]
    #             },
    #             input_csv_path=INPUT_CSV_PATH
    #         )
    #     except Exception as e:
    #         print(f"最终仿真出错: {e}")
    #         optimal_success = False
    #
    # if optimal_success:
    #     final_energy = simulator.total_energy / 3.6e6
    #     print(f"最优参数下的总能耗: {final_energy:.2f} kWh")
    #     simulator.save_results('optimal_simulation_results.csv')
    #     simulator.export_input_signals('optimal_input_signals.csv')
    #
    #     # 绘制关键结果曲线
    #     plt.figure(figsize=(10, 6))
    #     plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    #     plt.plot(
    #         simulator.results['time'],
    #         simulator.results['supplyTemp'] - 273.15,
    #         label='供水温度 [°C]',
    #         color='tab:blue'
    #     )
    #     plt.plot(
    #         simulator.results['time'],
    #         simulator.results['returnTemp'] - 273.15,
    #         label='回水温度 [°C]',
    #         color='tab:red'
    #     )
    #     plt.xlabel('时间 [s]')
    #     plt.ylabel('温度 [°C]')
    #     plt.title('最优制冷系统仿真结果')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig('optimal_results.png', dpi=150)
    #     plt.show()
    # else:
    #     print("最终仿真失败，无法生成结果文件")


if __name__ == "__main__":
    main()