import os
import re

from config import API_KEY, BASE_URL, MODEL_ID, TAVILY_API_KEY
from llm_client import OpenAICompatibleClient
from travel_tools import AGENT_SYSTEM_PROMPT, available_tools


def main() -> None:
    if not API_KEY or not BASE_URL or not MODEL_ID:
        print("请先在 config.py 中填写 API_KEY、BASE_URL、MODEL_ID。")
        return

    if TAVILY_API_KEY:
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

    llm = OpenAICompatibleClient(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
    prompt_history = [f"用户请求: {user_prompt}"]

    print(f"用户输入: {user_prompt}\n" + "=" * 40)

    for i in range(5):
        print(f"--- 循环 {i + 1} ---\n")

        full_prompt = "\n".join(prompt_history)
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        if llm_output.startswith("LLM_ERROR:"):
            print(f"模型调用失败: {llm_output}")
            print("本轮任务提前结束。请稍后重试，或更换可用模型。")
            break

        match = re.search(
            r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
            llm_output,
            re.DOTALL,
        )
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截断多余的 Thought-Action 对")

        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

        action_str = action_match.group(1).strip()
        if action_str.startswith("Finish"):
            finish_match = re.match(r"Finish\[(.*)\]", action_str)
            if finish_match:
                final_answer = finish_match.group(1)
                print(f"任务完成，最终答案: {final_answer}")
                break
            observation = "错误:Finish 格式不正确，应为 Finish[最终答案]"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

        tool_match = re.search(r"(\w+)\(", action_str)
        args_match = re.search(r"\((.*)\)", action_str)
        if not tool_match or not args_match:
            observation = "错误:无法解析工具名或参数，请使用 function_name(arg_name=\"arg_value\")"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

        tool_name = tool_match.group(1)
        args_str = args_match.group(1)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        if tool_name in available_tools:
            try:
                observation = available_tools[tool_name](**kwargs)
            except TypeError as e:
                observation = f"错误:工具参数不匹配 - {e}"
        else:
            observation = f"错误:未定义的工具 '{tool_name}'"

        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)


if __name__ == "__main__":
    main()
