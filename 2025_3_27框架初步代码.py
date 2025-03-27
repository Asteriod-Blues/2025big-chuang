# 完成时间：2025年3月17日
# 代码已经完成了框架的搭建，并且心理Agent的变量名称已经正确设置，选择器输出的流派为流派中文全称
# 选择器处理逻辑中已经包含了将中文流派名称到英文流派简称的一一对应功能
# 代码中除了选择器之外其他Agent的系统提示均未调整：包括各流派Agent、整合器、所有的评估器（评估标准）、整合管理agent
# 模型参数未调整，目前使用的硅基流动的API Key


import autogen
import json
import os
import re
from datetime import datetime

# 加载配置文件
config_list = autogen.config_list_from_json("autogen_try\\2_groupchat\\QAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# 用户代理
user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={"work_dir": "autogen_try\\other_example\\frame\\code", "use_docker": False}
)

# 选择器（修改系统提示）
selector = autogen.AssistantAgent(
    name="Selector",
    llm_config=llm_config,
    system_message="你是选择器，根据用户任务和之前的对话历史，从以下心理咨询流派中选择1-2个Agent：认知行为疗法、短焦疗法、精神分析疗法、叙事疗法、人本主义疗法。仅返回流派名称，用顿号分隔，例如 '认知行为疗法、人本主义疗法'，不要添加多余的换行符或重复内容。"
)

# 中英文流派名称映射
genre_mapping = {
    "认知行为疗法": "CBT",
    "短焦疗法": "SFT",
    "精神分析疗法": "PA",
    "叙事疗法": "NT",
    "人本主义疗法": "HT"
}

# 流派 Agents（使用英文简称）
genre_agents = {
    "CBT": autogen.AssistantAgent(name="CBTAgent", llm_config=llm_config, system_message="根据用户任务和之前的对话历史，生成神秘风格的文本。"),
    "SFT": autogen.AssistantAgent(name="SFTAgent", llm_config=llm_config, system_message="根据用户任务和之前的对话历史，生成奇幻风格的文本。"),
    "PA": autogen.AssistantAgent(name="PAAgent", llm_config=llm_config, system_message="根据用户任务和之前的对话历史，生成科幻风格的文本。"),
    "NT": autogen.AssistantAgent(name="NTAgent", llm_config=llm_config, system_message="根据用户任务和之前的对话历史，生成浪漫风格的文本。"),
    "HT": autogen.AssistantAgent(name="HTAgent", llm_config=llm_config, system_message="根据用户任务和之前的对话历史，生成历史风格的文本。")
}

# 评估器（用于流派文本评估，系统提示不变）
evaluators = {
    "Clarity": autogen.AssistantAgent(name="ClarityEvaluator", llm_config=llm_config, system_message="根据用户任务和对话历史，评估文本清晰度，返回0-10分，格式为：'评分：X分'（X为数字，可含小数）。"),
    "Creativity": autogen.AssistantAgent(name="CreativityEvaluator", llm_config=llm_config, system_message="根据用户任务和对话历史，评估文本创意性，返回0-10分，格式为：'评分：X分'（X为数字，可含小数）。"),
    "Coherence": autogen.AssistantAgent(name="CoherenceEvaluator", llm_config=llm_config, system_message="根据用户任务和对话历史，评估文本连贯性，返回0-10分，格式为：'评分：X分'（X为数字，可含小数）。"),
    "Emotion": autogen.AssistantAgent(name="EmotionEvaluator", llm_config=llm_config, system_message="根据用户任务和对话历史，评估文本情感强度，返回0-10分，格式为：'评分：X分'（X为数字，可含小数）。"),
    "Realism": autogen.AssistantAgent(name="RealismEvaluator", llm_config=llm_config, system_message="根据用户任务和对话历史，评估文本现实性，返回0-10分，格式为：'评分：X分'（X为数字，可含小数）。")
}

# 整合部分代理（系统提示不变）
text_integrator = autogen.AssistantAgent(
    name="TextIntegrator",
    llm_config=llm_config,
    system_message="你是文本整合器，根据两个流派的文本和评估分数，按连贯性和创意性标准整合成一段文本。如果收到管理 Agent 的修改意见，则根据意见优化文本并返回新版本。"
)

integration_evaluators = {
    "Consistency": autogen.AssistantAgent(
        name="ConsistencyEvaluator",
        llm_config=llm_config,
        system_message="根据用户任务和对话历史，评估整合文本的整体一致性（内容是否逻辑统一），返回0-10分和修改意见，格式为：'评分：X分\n修改意见：...'（X为数字，可含小数）。"
    ),
    "Engagement": autogen.AssistantAgent(
        name="EngagementEvaluator",
        llm_config=llm_config,
        system_message="根据用户任务和对话历史，评估整合文本的吸引力（是否引人入胜），返回0-10分和修改意见，格式为：'评分：X分\n修改意见：...'（X为数字，可含小数）。"
    ),
    "Fluency": autogen.AssistantAgent(
        name="FluencyEvaluator",
        llm_config=llm_config,
        system_message="根据用户任务和对话历史，评估整合文本的流畅性（语言是否自然流畅），返回0-10分和修改意见，格式为：'评分：X分\n修改意见：...'（X为数字，可含小数）。"
    )
}

integration_manager = autogen.AssistantAgent(
    name="IntegrationManager",
    llm_config=llm_config,
    system_message="你是整合管理 Agent，负责：1. 接收文本整合器的结果并传递给整合效果评估器；2. 收集评估结果，若平均分数≥8则输出最终文本，否则生成修改意见并传回文本整合器；3. 迭代最多5次，若仍未达标则选择最高分版本。"
)

# 初始化全局聊天历史和会话历史
all_chat_history = {}
conversation_history = []

# 选择器逻辑（添加中英文映射）
def selector_function(task, chat_history, conversation_history):
    context = "\n".join([f"用户: {entry['task']}\n框架: {entry['result']}" for entry in conversation_history])
    message = f"之前的对话历史：\n{context}\n\n根据当前任务选择1-2个流派Agent：\n{task}" if conversation_history else f"根据任务选择1-2个流派Agent：\n{task}"
    chat_result = user_proxy.initiate_chat(selector, message=message, max_turns=1)
    reply = chat_result.chat_history[-1]["content"].strip()
    print(f"选择器原始输出: {reply}")
    # 将中文流派名称映射为英文简称
    selected_genres_cn = [genre.strip() for genre in reply.split("、")]
    selected_genres = [genre_mapping.get(genre, "") for genre in selected_genres_cn if genre in genre_mapping]
    print(f"处理后的流派: {selected_genres}")
    chat_history["selector"] = chat_history.get("selector", []) + chat_result.chat_history
    return selected_genres

# 流派 Agent 并行生成文本
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

# 评估器并行评估（用于流派文本）
def run_evaluators(genre_results, chat_history, task, conversation_history):
    eval_results = {}
    context = "\n".join([f"用户: {entry['task']}\n框架: {entry['result']}" for entry in conversation_history])
    for i, result in enumerate(genre_results):
        genre_name = result.chat_history[-1]["name"].split(" -> ")[0] if result.chat_history else f"Genre_{i}"
        if not result.chat_history:
            print(f"警告：流派 {genre_name} 的 chat_history 为空，默认评估为 0 分。")
            eval_results[genre_name] = [
                autogen.ChatResult(chat_history=[{"content": "评分：0分", "role": "assistant"}])
                for _ in range(5)
            ]
            chat_history[f"evaluator_{genre_name}"] = chat_history.get(f"evaluator_{genre_name}", []) + eval_results[genre_name]
            continue
        text = result.chat_history[-1]["content"]
        eval_message = f"之前的对话历史：\n{context}\n\n当前用户任务：\n{task}\n\n评估以下文本：\n{text}" if conversation_history else f"当前用户任务：\n{task}\n\n评估以下文本：\n{text}"
        eval_queue = [
            {"sender": user_proxy, "recipient": evaluator, "message": eval_message, "max_turns": 1, "summary_method": "last_msg"}
            for evaluator in evaluators.values()
        ]
        eval_results[genre_name] = autogen.initiate_chats(eval_queue)
        chat_history[f"evaluator_{genre_name}"] = chat_history.get(f"evaluator_{genre_name}", []) + [eval.chat_history for eval in eval_results[genre_name]]
    print("评估结果：", eval_results.keys())
    return eval_results

# 从字符串中提取评分的函数
def extract_score(text):
    match = re.search(r'评分：(\d+\.?\d*)分', text)
    if match:
        return float(match.group(1))
    print(f"警告：无法从 '{text}' 中提取评分，默认返回 0。")
    return 0.0

# 整合逻辑（使用 Nested Chat）
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
            threshold = 8.0

            for iteration in range(max_iterations):
                if iteration == 0:
                    message = initial_message
                else:
                    feedback = "\n".join([f"{key}: {eval_result['score']}分, 修改意见: {eval_result['suggestion']}" for key, eval_result in last_eval_results.items()])
                    message = f"之前的文本：\n{current_text}\n\n评估反馈：\n{feedback}\n\n请根据反馈优化文本。"
                
                integration_result = manager.initiate_chat(
                    integrator,
                    message=message,
                    max_turns=1
                )
                current_text = integration_result.chat_history[-1]["content"]
                chat_history[f"integration_text_iter_{iteration+1}"] = integration_result.chat_history

                context = "\n".join([f"用户: {entry['task']}\n框架: {entry['result']}" for entry in conversation_history])
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

# 主流程
def main():
    global all_chat_history, conversation_history
    output_dir = "autogen_try\\other_example\\frame"
    os.makedirs(output_dir, exist_ok=True)
    chat_file = f"{output_dir}\\chat_history11.json"

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

        if len(selected_genres) == 2:
            eval_results = run_evaluators(genre_results, current_chat_history, task, conversation_history)
        else:
            eval_results = None

        final_result = integrate_results(selected_genres, genre_results, eval_results, current_chat_history, conversation_history, task)
        print(f"\n最终结果:\n{final_result}")

        conversation_history.append({"task": task, "result": final_result})

        with open(f"{output_dir}\\result11.json", "w", encoding="utf-8") as f:
            json.dump({"task": task, "selected_genres": selected_genres, "final_result": final_result}, f, ensure_ascii=False, indent=4)
        
        with open(chat_file, "w", encoding="utf-8") as f:
            json.dump(all_chat_history, f, ensure_ascii=False, indent=4)
        print(f"聊天历史已保存到 {chat_file}")
        
        round_num += 1

if __name__ == "__main__":
    main()