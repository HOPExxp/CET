#-*-coding:utf-8-*-
# -*-coding:utf-8-*-
import os
import matplotlib.pyplot as plt
import numpy as np
from random_walk_optimizer import RandomWalkOptimizer  # 替换为随机走步优化器
from cooling_system_simulator import CoolingSystemSimulator
import warnings
from contextlib import redirect_stderr

warnings.filterwarnings("ignore")


def main():
    """
    主函数：集成随机走步优化和制冷系统仿真器，优化Tset和coolingLoad参数
    """
    # 1. 配置文件路径
    FMU_PATH = r"C:\Users\xxp\Documents\Dymola\CET-zhuzhouyidong\simulation_changeinput\CoolingStation_0changeinput_System_Coolingstation.fmu"
    INPUT_CSV_PATH = "input_data.csv"  # 其他参数从该CSV文件导入

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
        x: 参数向量，格式为 [Tset, coolingLoad]
        """
        tset_value, cooling_load_value = x

        # 转换为更易读的单位显示
        print(f"尝试 Tset={tset_value-273.15:.2f}°C, coolingLoad={cooling_load_value/1e6:.2f}MW")

        # 运行仿真
        with open(os.devnull, 'w') as f, redirect_stderr(f):
            try:
                success = simulator.simulate(
                    start_time=start_time,
                    stop_time=stop_time,
                    step_size=step_size,
                    input_values={
                        'Tset': tset_value,
                        'coolingLoad': cooling_load_value
                    },
                    input_csv_path=INPUT_CSV_PATH
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
            return 1e10

    # 5. 仿真与优化参数配置
    start_time = 0.0
    stop_time = 7200.0  # 2小时仿真
    step_size = 30.0  # 优化阶段步长

    # 优化变量配置（Tset和coolingLoad）
    optimization_vars = [
        {
            "name": "Tset",
            "bounds": (273.15 + 16, 273.15 + 20),  # 16-20°C（开尔文）
            "unit": "K",
            "scale": 1.0,
            "scale_unit": "°C",
            "offset": 273.15
        },
        {
            "name": "coolingLoad",
            "bounds": (5e6, 10e6),  # 5-10MW（瓦）
            "unit": "W",
            "scale": 1e-6,
            "scale_unit": "MW",
            "offset": 0
        }
    ]
    bounds = [var["bounds"] for var in optimization_vars]

    # 6. 运行随机走步优化
    print("\n开始随机走步优化...")
    rw_optimizer = RandomWalkOptimizer(
        objective_function=objective_function,
        bounds=bounds,
        initial_step_size=0.1,  # 初始步长比例（相对于参数范围）
        step_decay=0.95,  # 步长衰减系数
        max_iterations=30,  # 迭代次数
        num_walkers=5  # 并行走步数量
    )

    best_solution, best_fitness = rw_optimizer.optimize()

    # 7. 绘制优化历史
    rw_optimizer.plot_optimization_history('rw_optimization_history.png')

    # 8. 输出优化结果
    print(f"\n优化完成!")
    for i, var in enumerate(optimization_vars):
        value = best_solution[i]
        display_value = value * var["scale"] + var["offset"]
        print(f"最优 {var['name']}: {display_value:.2f} {var['scale_unit']} ({value:.2f} {var['unit']})")
    print(f"最小能耗: {best_fitness:.2f} kWh")

    # 9. 最终精确仿真
    print("\n使用最优参数运行最终仿真...")
    final_step_size = 5.0

    with open(os.devnull, 'w') as f, redirect_stderr(f):
        try:
            optimal_success = simulator.simulate(
                start_time=start_time,
                stop_time=stop_time,
                step_size=final_step_size,
                input_values={
                    'Tset': best_solution[0],
                    'coolingLoad': best_solution[1]
                },
                input_csv_path=INPUT_CSV_PATH
            )
        except Exception as e:
            print(f"最终仿真出错: {e}")
            optimal_success = False

    if optimal_success:
        final_energy = simulator.total_energy / 3.6e6
        print(f"最优参数下的总能耗: {final_energy:.2f} kWh")
        simulator.save_results('optimal_simulation_results.csv')
        simulator.export_input_signals('optimal_input_signals.csv')

        # 绘制关键结果曲线
        plt.figure(figsize=(10, 6))
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
        plt.savefig('optimal_results.png', dpi=150)
        plt.show()
    else:
        print("最终仿真失败，无法生成结果文件")


if __name__ == "__main__":
    main()