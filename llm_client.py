import time

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError


class OpenAICompatibleClient:
    """
    一个用于调用任何兼容 OpenAI 接口的 LLM 服务客户端。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=0)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用 LLM API 生成回应。"""
        print("正在调用大语言模型...")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                )
                answer = response.choices[0].message.content
                print("大语言模型响应成功。")
                return answer or ""
            except RateLimitError as e:
                print(f"调用 LLM API 限流(第 {attempt}/{max_attempts} 次): {e}")
                if attempt == max_attempts:
                    return f"LLM_ERROR:RATE_LIMIT:{e}"
                time.sleep(2 * attempt)
            except (APITimeoutError, APIConnectionError) as e:
                print(f"调用 LLM API 网络/超时错误(第 {attempt}/{max_attempts} 次): {e}")
                if attempt == max_attempts:
                    return f"LLM_ERROR:NETWORK:{e}"
                time.sleep(2 * attempt)
            except Exception as e:
                print(f"调用 LLM API 时发生错误: {e}")
                return f"LLM_ERROR:UNKNOWN:{e}"
