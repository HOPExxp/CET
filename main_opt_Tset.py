#-*-coding:utf-8-*-
#从main_opt_coolingLoad改过来的
import os
import matplotlib.pyplot as plt
import numpy as np
from genetic_algorithm import GeneticAlgorithmOptimizer
from cooling_system_simulator import CoolingSystemSimulator
import warnings
import sys
from contextlib import redirect_stderr

# 禁用警告以保持输出整洁
warnings.filterwarnings("ignore")


def main():
    """
    主函数：集成遗传算法和制冷系统仿真器，仅优化Tset参数，其他参数从CSV导入
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

    # 4. 定义目标函数（仅优化Tset）
    def objective_function(x):
        """
        目标函数：计算给定Tset下的系统能耗

        Parameters:
        x: 参数向量，格式为 [Tset]（单位：开尔文）

        Returns:
        系统总能耗（kWh），值越小越优
        """
        tset_value = x[0]
        # 转换为摄氏度显示（开尔文转摄氏度：-273.15）
        print(f"尝试 Tset={tset_value-273.15:.2f}°C")

        # 运行仿真（重定向stderr避免冗余输出）
        with open(os.devnull, 'w') as f, redirect_stderr(f):
            try:
                success = simulator.simulate(
                    start_time=start_time,
                    stop_time=stop_time,
                    step_size=step_size,
                    input_values={'Tset': tset_value},  # 仅传递待优化参数Tset
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
    # 统一仿真时间参数（确保优化与最终验证的能耗基准一致）
    start_time = 0.0
    stop_time = 7200.0  # 2小时仿真（与最终验证保持一致）
    step_size = 30.0  # 优化阶段步长（兼顾速度）

    # 优化变量配置（仅Tset）
    optimization_vars = [
        {
            "name": "Tset",
            "bounds": (273.15 + 16, 273.15 + 20),  # Tset范围：16-20°C（转换为开尔文）
            "unit": "K",
            "scale": 1.0,
            "scale_unit": "°C",
            "offset": 273.15  # 开尔文到摄氏度的转换偏移量
        }
    ]
    # 提取边界用于遗传算法
    bounds = [var["bounds"] for var in optimization_vars]

    # 6. 运行遗传算法优化
    print("\n开始遗传算法优化...")
    ga_optimizer = GeneticAlgorithmOptimizer(
        objective_function=objective_function,
        bounds=bounds,
        population_size=10,  # 适当增大种群提高搜索能力
        max_generations=10,  # 增加迭代次数确保收敛
        mutation_rate=0.15,
        crossover_rate=0.8
    )

    best_solution, best_fitness = ga_optimizer.optimize()

    # 7. 绘制优化历史（已处理中文显示）
    ga_optimizer.plot_optimization_history('.\数据文件\Output\ga_optimization_history.png')

    # 8. 输出优化结果
    print(f"\n优化完成!")
    for i, var in enumerate(optimization_vars):
        value = best_solution[i]
        # 转换为摄氏度显示
        display_value = value - var["offset"]
        print(f"最优 {var['name']}: {display_value:.2f} {var['scale_unit']} ({value:.2f} {var['unit']})")
    print(f"最小能耗: {best_fitness:.2f} kWh")

    # 9. 最终精确仿真（使用优化结果）
    print("\n使用最优参数运行最终仿真...")
    final_step_size = 5.0  # 更小步长提高精度

    with open(os.devnull, 'w') as f, redirect_stderr(f):
        try:
            optimal_success = simulator.simulate(
                start_time=start_time,
                stop_time=stop_time,  # 与优化阶段相同的时间
                step_size=step_size,
                input_values={'Tset': best_solution[0]},
                input_csv_path=INPUT_CSV_PATH
            )
        except Exception as e:
            print(f"最终仿真出错: {e}")
            optimal_success = False

    if optimal_success:
        final_energy = simulator.total_energy / 3.6e6
        print(f"最优参数下的总能耗: {final_energy:.2f} kWh")
        simulator.save_results('.\数据文件\Output\optimal_simulation_results.csv')
        simulator.export_input_signals('.\数据文件\Output\optimal_input_signals.csv')

        # 绘制关键结果曲线
        plt.figure(figsize=(10, 6))
        # 设置中文字体
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.plot(
            simulator.results['time'],
            simulator.results['supplyTemp'] - 273.15,
            label='供水温度 [°C]',
            color='tab:blue'
        )
        plt.plot(
            simulator.results['time'],
            simulator.results['returnTemp'] - 273.15,
            label='回水温度 [°C]',
            color='tab:red'
        )
        plt.xlabel('时间 [s]')
        plt.ylabel('温度 [°C]')
        plt.title('最优制冷系统仿真结果')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('.\数据文件\Output\optimal_results.png', dpi=150)
        plt.show()
    else:
        print("最终仿真失败，无法生成结果文件")


if __name__ == "__main__":
    main()
