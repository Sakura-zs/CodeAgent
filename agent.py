import subprocess
import time
import json
import os
from collections import deque
from openai import OpenAI

# ==========================================
# 1. 配置你的大模型 (支持 OpenAI, 通义千问, DeepSeek 等)
# ==========================================
client = OpenAI(
    api_key="xxx",  # 填入你的 API Key
    base_url="https://api.deepseek.com" # 如果用国内大模型，填入对应的 base_url
)
MODEL_NAME = "deepseek-v4-flash" # 或 "deepseek-chat", "qwen-plus" 等

# ==========================================
# 2. 训练环境管理器
# ==========================================
class TrainingEnvironment:
    def __init__(self, log_file="training.log"):
        self.process = None
        self.log_file = log_file

    def start_job(self, command):
        """后台启动训练任务，并重定向输出到日志文件"""
        print(f"\n🚀 [Agent 执行] 启动命令: {command}")
        # 打开文件准备写入日志
        self.file_handle = open(self.log_file, "w", encoding="utf-8")
        self.process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=self.file_handle, 
            stderr=subprocess.STDOUT
        )

    def get_status_and_logs(self, lines=50):
        """检查进程状态，并读取最后几行日志"""
        if self.process is None:
            return "NO_JOB", "没有正在运行的任务。"

        ret_code = self.process.poll()
        
        # 使用 deque 高效读取文件的最后 N 行
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                recent_logs = "".join(deque(f, maxlen=lines))
        except Exception as e:
            recent_logs = f"读取日志失败: {e}"

        if ret_code is None:
            return "RUNNING", recent_logs
        elif ret_code == 0:
            return "SUCCESS", recent_logs
        else:
            return "FAILED", recent_logs

    def kill_job(self):
        """强制终止当前任务"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
            print("⚠️ [Agent 执行] 任务已被强制终止。")

# ==========================================
# 3. 大模型决策函数
# ==========================================
def ask_llm_for_decision(status, logs):
    """把终端日志发给 LLM，让它决定下一步怎么做"""
    
    # 如果任务已经自己结束了，就不需要 LLM 决策终止了
    if status == "SUCCESS":
        return {"decision": "next", "reason": "训练正常结束，退出码为 0"}
    
    prompt = f"""
    你是一个机器学习训练监工。当前任务状态是: {status}。
    以下是终端输出的最后 50 行日志：
    ```text
    {logs}
    ```
    
    请分析日志，并做出决策：
    1. 如果模型正常训练（loss在下降，或者正常打印epoch），请选择 "continue"。
    2. 如果发生严重错误（如 Out of Memory, Loss 变为 NaN, 文件找不到等导致训练彻底卡死或崩溃），请选择 "kill"。
    3. 如果训练已经报错退出（状态为 FAILED），请选择 "next" 准备跑下一个。

    请必须以 JSON 格式返回，包含 'decision' (只能是 continue, kill, next 之一) 和 'reason' (你做出决策的原因)。
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={ "type": "json_object" }, # 强制输出 JSON
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"❌ [解析 LLM 回复失败]: {e}")
        return {"decision": "continue", "reason": "解析失败，默认继续观察"}

# ==========================================
# 4. 主控循环 (The Agent Loop)
# ==========================================
def main():
    # 这里写死你要跑的参数组合，也可以从文件里读
    tasks = [
        "python mock_train.py --lr 0.01",   # 假设这个会正常跑完
        "python mock_train.py --lr NaN",    # 假设这个会导致 loss 爆炸
        "python mock_train.py --lr 0.001"   # 假设这个正常
    ]

    env = TrainingEnvironment()

    for idx, task_cmd in enumerate(tasks):
        print(f"\n======================================")
        print(f"⚙️ 开始执行任务 {idx + 1}/{len(tasks)}")
        print(f"======================================")
        
        env.start_job(task_cmd)
        
        while True:
            # 监控频率：真实训练中建议设为 300 秒 (5分钟)，测试时设为 5 秒
            time.sleep(5) 
            
            # 1. 拿日志
            status, logs = env.get_status_and_logs(lines=30)
            print(f"👀 [Agent 观察] 进程状态: {status} | 正在呼叫大模型分析日志...")
            
            # 2. 问大模型
            decision_data = ask_llm_for_decision(status, logs)
            decision = decision_data.get('decision', 'continue')
            reason = decision_data.get('reason', '未知')
            
            print(f"🧠 [LLM 决策]: {decision.upper()} | 理由: {reason}")
            
            # 3. 采取行动
            if decision == "continue":
                # 什么都不做，继续等
                pass
            
            elif decision == "kill":
                env.kill_job()
                print("⏭️ 准备切换到下一个训练参数...")
                break # 跳出 while 循环，进入外层 for 循环的下一个 task
                
            elif decision == "next":
                if status == "RUNNING":
                    env.kill_job() # 防止 LLM 误判，确保进程死掉
                print("✅ 当前任务结束，准备跑下一个...")
                break

    print("\n🎉 所有训练任务执行完毕！Agent 下班！")

if __name__ == "__main__":
    main()
