# 2025_4_15框架代码修改第1版
# 将打分器的回答先传送给了整合管理agent，后让这个agent将历史记录和评分传给整合器
# 打分器逻辑修改成了：理论连贯性、目标一致性、技术兼容性
# 其中理论连贯性打分器针对不同的流派分别设计，只会对特定流派的回答打分，共5个，其他打分器数量都为1个
# 理论连贯性、目标一致性打分器的分数会传至技术兼容性的打分器中，作为参考


import autogen
import json
import os
import re
from datetime import datetime
from ollama import Client as OllamaClient

# 自定义 Ollama 客户端（保持不变）
class CustomOllamaClient:
    def __init__(self, config):
        self.client = OllamaClient(host=config["base_url"])
        self.model = config["model"]

    def create(self, params):
        messages = params.get("messages", [])
        response = self.client.chat(
            model=self.model,
            messages=messages
        )
        return {
            "choices": [{"message": {"content": response["message"]["content"]}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

# 加载配置文件（保持不变）
config_list = autogen.config_list_from_json("autogen_try\\2_groupchat\\QAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}
llm_config2 = {
    "config_list": [
        {
            "model": "deepseek-r1:1.5b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "dummy_key"
        }
    ]
}

# 用户代理（保持不变）
user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={"work_dir": "autogen_try\\other_example\\frame\\code", "use_docker": False}
)

# 选择器（保持不变）
selector = autogen.ConversableAgent(
    name="Selector",
    llm_config=llm_config2,
    system_message="你是选择器，根据用户任务和之前的对话历史，从以下心理咨询流派中选择1-2个Agent：认知行为疗法、短焦疗法、精神分析疗法、叙事疗法、人本主义疗法。仅返回流派名称，用顿号分隔，例如 '认知行为疗法、人本主义疗法'，不要添加多余的换行符或重复内容。"
)

# 中英文流派名称映射（保持不变）
genre_mapping = {
    "认知行为疗法": "CBT",
    "短焦疗法": "SFT",
    "精神分析疗法": "PA",
    "叙事疗法": "NT",
    "人本主义疗法": "HT"
}

# 流派 Agents（保持不变）
genre_agents = {
    "CBT": autogen.ConversableAgent(name="CBTAgent", llm_config=llm_config2, system_message="根据用户任务和之前的对话历史，生成神秘风格的文本。"),
    "SFT": autogen.ConversableAgent(name="SFTAgent", llm_config=llm_config2, system_message="根据用户任务和之前的对话历史，生成奇幻风格的文本。"),
    "PA": autogen.ConversableAgent(name="PAAgent", llm_config=llm_config2, system_message="根据用户任务和之前的对话历史，生成科幻风格的文本。"),
    "NT": autogen.ConversableAgent(name="NTAgent", llm_config=llm_config2, system_message="根据用户任务和之前的对话历史，生成浪漫风格的文本。"),
    "HT": autogen.ConversableAgent(name="HTAgent", llm_config=llm_config2, system_message="根据用户任务和之前的对话历史，生成历史风格的文本。")
}

# 动态生成评估器（保持不变）
def create_evaluators(selected_genres):
    evaluators = {}
    for genre in selected_genres:
        evaluators[f"TheoreticalCoherence_{genre}"] = autogen.AssistantAgent(
            name=f"TheoreticalCoherenceEvaluator_{genre}",
            llm_config=llm_config,
            system_message=f"根据用户任务和对话历史，评估 {genre} 流派生成文本的理论连贯性（内容是否符合 {genre} 流派的理论框架），返回0-5分，格式为：'评分：X分'（X为数字，可含小数）。"
        )
    evaluators["GoalConsistency"] = autogen.AssistantAgent(
        name="GoalConsistencyEvaluator",
        llm_config=llm_config,
        system_message="根据用户任务和对话历史，评估文本的目标一致性（内容是否与用户任务目标一致），返回0-5分，格式为：'评分：X分'（X为数字，可含小数）。"
    )
    evaluators["TechniqueCompatibility"] = autogen.AssistantAgent(
        name="TechniqueCompatibilityEvaluator",
        llm_config=llm_config,
        system_message="根据用户任务和对话历史，参考理论连贯性和目标一致性分数，评估文本的技术兼容性（内容是否符合心理咨询流派的技术方法），返回0-5分，格式为：'评分：X分'（X为数字，可含小数）。"
    )
    return evaluators

# 整合部分代理（保持不变）
text_integrator = autogen.AssistantAgent(
    name="TextIntegrator",
    llm_config=llm_config,
    system_message="你是文本整合器，根据两个流派的文本和评估分数，按连贯性和创意性标准整合成一段文本。如果收到管理 Agent 的修改意见，则根据意见优化文本并返回新版本。"
)

# 整合效果评估器（保持不变）
integration_evaluators = {
    "TheoreticalCoherence": autogen.AssistantAgent(
        name="TheoreticalCoherenceEvaluator",
        llm_config=llm_config,
        system_message="根据用户任务和对话历史，评估整合文本的理论连贯性（内容是否符合心理咨询流派的理论框架），返回0-5分和修改意见，格式为：'评分：X分\n修改意见：...'（X为数字，可含小数）。"
    ),
    "GoalConsistency": autogen.AssistantAgent(
        name="GoalConsistencyEvaluator",
        llm_config=llm_config,
        system_message="根据用户任务和对话历史，评估整合文本的目标一致性（内容是否与用户任务目标一致），返回0-5分和修改意见，格式为：'评分：X分\n修改意见：...'（X为数字，可含小数）。"
    ),
    "TechniqueCompatibility": autogen.AssistantAgent(
        name="TechniqueCompatibilityEvaluator",
        llm_config=llm_config,
        system_message="根据用户任务和对话历史，评估整合文本的技术兼容性（内容是否符合心理咨询流派的技术方法），返回0-5分和修改意见，格式为：'评分：X分\n修改意见：...'（X为数字，可含小数）。"
    )
}

integration_manager = autogen.AssistantAgent(
    name="IntegrationManager",
    llm_config=llm_config,
    system_message="你是整合管理 Agent，负责：1. 接收文本整合器的结果并传递给整合效果评估器；2. 收集评估结果和流派文本，将其传递给文本整合器；3. 迭代最多5次，若平均分数≥4则输出最终文本，否则继续优化；4. 若仍未达标则选择最高分版本。"
)

# 初始化全局聊天历史和会话历史（保持不变）
all_chat_history = {}
conversation_history = []

# 选择器逻辑（保持不变）
def selector_function(task, chat_history, conversation_history):
    context = "\n".join([f"用户: {entry['task']}\n框架: {entry['result']}" for entry in conversation_history])
    message = f"之前的对话历史：\n{context}\n\n根据当前任务选择1-2个流派Agent：\n{task}" if conversation_history else f"根据任务选择1-2个流派Agent：\n{task}"
    chat_result = user_proxy.initiate_chat(selector, message=message, max_turns=1)
    reply = chat_result.chat_history[-1]["content"].strip()
    print(f"选择器原始输出: {reply}")
    cleaned_reply = reply.split('\n')[-1].strip() if '<think>' in reply else reply
    selected_genres_cn = [genre.strip() for genre in cleaned_reply.split("、")]
    selected_genres = [genre_mapping.get(genre) for genre in selected_genres_cn if genre_mapping.get(genre)]
    print(f"处理后的流派: {selected_genres}")
    chat_history["selector"] = chat_history.get("selector", []) + chat_result.chat_history
    return selected_genres

# 流派 Agent 并行生成文本（保持不变）
def run_genre_agents(selected_genres, task, chat_history, conversation_history):
    valid_genres = [genre for genre in selected_genres if genre in genre_agents]
    if not valid_genres:
        print("错误：没有有效的流派名称，跳过生成。")
        return []
    context = "\n".join([f"用户: {entry['task']}\n框架: {entry['result']}" for entry in conversation_history])
    message = f"之前的对话历史：\n{context}\n\n当前任务：\n{task}" if conversation_history else f"当前任务：\n{task}"
    chat_queue = [
        {"sender": user_proxy, "recipient": genre_agents[genre], "message": message, "max_turns": 1, "summary_method": "last_msg"}
        for genre in valid_genres
    ]
    results = autogen.initiate_chats(chat_queue)
    print("流派生成结果：")
    for i, result in enumerate(results):
        print(f"流派 {valid_genres[i]}: {result.chat_history}")
        chat_history[f"genre_{valid_genres[i]}"] = chat_history.get(f"genre_{valid_genres[i]}", []) + result.chat_history
    return results

# 修改后的评估器串行逻辑
def run_evaluators(genre_results, chat_history, task, conversation_history, selected_genres):
    eval_results = {}
    context = "\n".join([f"用户: {entry['task']}\n框架: {entry['result']}" for entry in conversation_history])
    evaluators = create_evaluators(selected_genres)
    print(f"生成的评估器: {list(evaluators.keys())}")

    for i, result in enumerate(genre_results):
        genre_name = result.chat_history[-1]["name"].split(" -> ")[0] if result.chat_history else f"Genre_{i}"
        if not result.chat_history:
            print(f"警告：流派 {genre_name} 的 chat_history 为空，默认评估为 0 分。")
            eval_results[genre_name] = [
                {"content": "评分：0分", "role": "assistant"},
                {"content": "评分：0分", "role": "assistant"},
                {"content": "评分：0分", "role": "assistant"}
            ]
            chat_history[f"evaluator_{genre_name}"] = chat_history.get(f"evaluator_{genre_name}", []) + eval_results[genre_name]
            continue
        
        text = result.chat_history[-1]["content"]
        base_message = f"之前的对话历史：\n{context}\n\n当前用户任务：\n{task}\n\n评估以下 {genre_name} 流派生成的文本：\n{text}" if conversation_history else f"当前用户任务：\n{task}\n\n评估以下 {genre_name} 流派生成的文本：\n{text}"
        
        # 步骤 1：串行执行 TheoreticalCoherence 评估
        theo_key = f"TheoreticalCoherence_{selected_genres[i]}"
        theo_evaluator = evaluators[theo_key]
        theo_result = user_proxy.initiate_chat(
            theo_evaluator,
            message=base_message,
            max_turns=1
        )
        theo_score = extract_score(theo_result.chat_history[-1]["content"])
        print(f"流派 {genre_name} 的 {theo_key} 评分: {theo_result.chat_history[-1]['content']}")

        # 步骤 2：并行执行 GoalConsistency 评估
        goal_evaluator = evaluators["GoalConsistency"]
        goal_result = user_proxy.initiate_chat(
            goal_evaluator,
            message=base_message,
            max_turns=1
        )
        goal_score = extract_score(goal_result.chat_history[-1]["content"])
        print(f"流派 {genre_name} 的 GoalConsistency 评分: {goal_result.chat_history[-1]['content']}")

        # 步骤 3：串行执行 TechniqueCompatibility 评估，参考前两者的分数
        tech_evaluator = evaluators["TechniqueCompatibility"]
        tech_message = f"{base_message}\n\n参考评分：\n- 理论连贯性: {theo_score}分\n- 目标一致性: {goal_score}分"
        tech_result = user_proxy.initiate_chat(
            tech_evaluator,
            message=tech_message,
            max_turns=1
        )
        tech_score = extract_score(tech_result.chat_history[-1]["content"])
        print(f"流派 {genre_name} 的 TechniqueCompatibility 评分: {tech_result.chat_history[-1]['content']}")

        # 步骤 4：收集结果
        eval_results[genre_name] = [
            {"content": theo_result.chat_history[-1]["content"], "role": "assistant"},
            {"content": goal_result.chat_history[-1]["content"], "role": "assistant"},
            {"content": tech_result.chat_history[-1]["content"], "role": "assistant"}
        ]
        chat_history[f"evaluator_{genre_name}"] = chat_history.get(f"evaluator_{genre_name}", []) + [theo_result.chat_history, goal_result.chat_history, tech_result.chat_history]
    
    print("评估结果：", eval_results.keys())
    return eval_results

# 从字符串中提取评分的函数（保持不变）
def extract_score(text):
    match = re.search(r'评分：(\d+\.?\d*)分', text)
    if match:
        return float(match.group(1))
    print(f"警告：无法从 '{text}' 中提取评分，默认返回 0。")
    return 0.0

# 整合逻辑（保持不变）
def integrate_results(selected_genres, genre_results, eval_results, chat_history, conversation_history, task):
    if len(selected_genres) == 1:
        if not genre_results or not genre_results[0].chat_history:
            return "错误：流派生成失败，无可用文本。"
        final_text = genre_results[0].chat_history[-1]["content"]
        chat_history["integration"] = [{"content": f"单一流派结果：{final_text}", "role": "assistant", "name": "IntegrationManager"}]
        return final_text
    else:
        if not eval_results:
            return "错误：评估失败，无法整合文本。"
        
        text1 = genre_results[0].chat_history[-1]["content"] if genre_results and genre_results[0].chat_history else "无文本"
        text2 = genre_results[1].chat_history[-1]["content"] if len(genre_results) > 1 and genre_results[1].chat_history else "无文本"
        scores = {}
        for genre in selected_genres:
            if genre not in eval_results or not eval_results[genre]:
                scores[genre] = 0
            else:
                scores[genre] = sum(extract_score(eval.chat_history[-1]["content"]) for eval in eval_results[genre])
        
        context = "\n".join([f"用户: {entry['task']}\n框架: {entry['result']}" for entry in conversation_history])
        initial_message = f"之前的对话历史：\n{context}\n\n当前用户任务：\n{task}\n\n整合以下两个流派文本：\n1. {text1}\n2. {text2}\n评分: {scores}"

        def nested_integration(manager, integrator, evaluators, initial_message, chat_history, task, conversation_history):
            versions = []
            current_text = None
            max_iterations = 5
            threshold = 4.0

            for iteration in range(max_iterations):
                if iteration == 0:
                    message_to_integrator = initial_message
                else:
                    feedback = "\n".join([f"{key}: {eval_result['score']}分, 修改意见: {eval_result['suggestion']}" for key, eval_result in last_eval_results.items()])
                    message_to_integrator = f"之前的对话历史：\n{context}\n\n当前用户任务：\n{task}\n\n之前整合的文本：\n{current_text}\n\n流派文本：\n1. {text1}\n2. {text2}\n\n评估反馈：\n{feedback}\n\n请根据以上信息优化整合文本。"

                integration_result = manager.initiate_chat(
                    integrator,
                    message=message_to_integrator,
                    max_turns=1
                )
                current_text = integration_result.chat_history[-1]["content"]
                chat_history[f"integration_text_iter_{iteration+1}"] = integration_result.chat_history

                eval_message = f"之前的对话历史：\n{context}\n\n当前用户任务：\n{task}\n\n评估以下整合文本：\n{current_text}" if conversation_history else f"当前用户任务：\n{task}\n\n评估以下整合文本：\n{current_text}"
                eval_queue = [
                    {"sender": manager, "recipient": evaluator, "message": eval_message, "max_turns": 1, "summary_method": "last_msg"}
                    for evaluator in evaluators.values()
                ]
                eval_results = autogen.initiate_chats(eval_queue)
                chat_history[f"integration_eval_iter_{iteration+1}"] = [result.chat_history for result in eval_results]

                last_eval_results = {}
                for i, (key, evaluator) in enumerate(evaluators.items()):
                    eval_text = eval_results[i].chat_history[-1]["content"]
                    score = extract_score(eval_text)
                    suggestion = eval_text.split("修改意见：")[-1] if "修改意见：" in eval_text else "无"
                    last_eval_results[key] = {"score": score, "suggestion": suggestion}

                avg_score = sum(result["score"] for result in last_eval_results.values()) / len(last_eval_results)
                versions.append({"text": current_text, "avg_score": avg_score})

                if avg_score >= threshold:
                    chat_history["integration_final"] = [{"content": f"最终整合文本：{current_text} (平均分数: {avg_score})", "role": "assistant", "name": "IntegrationManager"}]
                    return current_text

            best_version = max(versions, key=lambda x: x["avg_score"])
            chat_history["integration_final"] = [{"content": f"达到最大迭代次数，最终选择最高分版本：{best_version['text']} (平均分数: {best_version['avg_score']})", "role": "assistant", "name": "IntegrationManager"}]
            return best_version["text"]

        final_text = nested_integration(
            integration_manager,
            text_integrator,
            integration_evaluators,
            initial_message,
            chat_history,
            task,
            conversation_history
        )
        return final_text

# 主流程（保持不变）
def main():
    global all_chat_history, conversation_history
    output_dir = "autogen_try\\other_example\\frame"
    os.makedirs(output_dir, exist_ok=True)
    chat_file = f"{output_dir}\\chat_history14.json"

    print("欢迎使用多轮交互框架！输入任务开始，输入 'exit' 退出。")
    round_num = 1
    while True:
        task = input("\n请输入任务：")
        if task.lower() == "exit":
            print("退出程序。")
            break

        current_chat_history = {}
        all_chat_history[f"round_{round_num}"] = current_chat_history

        selected_genres = selector_function(task, current_chat_history, conversation_history)
        print(f"选择的流派: {selected_genres}")

        genre_results = run_genre_agents(selected_genres, task, current_chat_history, conversation_history)

        if len(selected_genres) >= 1:
            eval_results = run_evaluators(genre_results, current_chat_history, task, conversation_history, selected_genres)
        else:
            eval_results = None

        final_result = integrate_results(selected_genres, genre_results, eval_results, current_chat_history, conversation_history, task)
        print(f"\n最终结果:\n{final_result}")

        conversation_history.append({"task": task, "result": final_result})

        with open(f"{output_dir}\\result14.json", "w", encoding="utf-8") as f:
            json.dump({"task": task, "selected_genres": selected_genres, "final_result": final_result}, f, ensure_ascii=False, indent=4)
        
        with open(chat_file, "w", encoding="utf-8") as f:
            json.dump(all_chat_history, f, ensure_ascii=False, indent=4)
        print(f"聊天历史已保存到 {chat_file}")
        
        round_num += 1

if __name__ == "__main__":
    main()