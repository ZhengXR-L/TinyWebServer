Agent基本概念
Environment+Perception->Agent->Action
自主性
定义：以大语言模型为大脑驱动，具有自主理解感知、规划、记忆和使用工具的能力，能够自动化执行完成复杂任务的系统。
Agent的主要组成模块：记忆Memory、工具Tools、规划Planning和行动Action（提示词Prompt）
● 记忆：短期记忆和长期记忆。短期记忆（一般在内存）：上下文、对话历史；长期记忆（一般在数据库）：偏好、行业的知识。Memory Agent
● 规划：事前规划before和事后反思After。先规划再执行、边走边看；思维链、子目标分解。
● 工具：API/MCP等。日历、计算器、代码解释器和搜索
● 行动：Agent的行为。Π(θ)
例子：用户想要五一出行的一个旅游规划（输入）。用户给定的相关信息，比如预算、人数、地点等信息就是短期记忆，而Agent根据用户之前的使用，记录了一些喜好在数据库里，根据这些喜好来做推荐，这部分知识就是长期记忆。Agent做计划时需要知道酒店、景区等地方的价格，这部分信息在数据库没有，便可以通过工具（搜索）去查询。Agent可以把用户给旅游规划这个高层级的任务拆分成几个小目标，比如：确认出行工具，查询目的地信息，进行行程规划，订票。比如用户觉得预算太高，Agent就会把预算作为一项约束，重新规划，进行动态修正（可能给出错误或者让用户不满的方案）。
Agent和Workflow
核心不同：Workflow是按部就班完成目标；Agent具有高度的自主性。
WorkFlow：优点：精确的、静态的、可控的、速度快；缺点：不灵活，解决的问题较为简单
Agent：有不确定性、以大模型为中枢
动手实现的一个Agent
准备工作：
使用大模型API（千问）
https://help.aliyun.com/zh/model-studio/get-api-key
调用大模型需要的配置
● Base URL
● API Key
● Model ID

使用TAVILY：
Tavily 是一款专为 AI 智能体（AI Agent）和大型语言模型（LLM）设计的搜索引擎 API
https://app.tavily.com/home

Tavily 对开发者非常友好，注册流程简单（通常无需绑定信用卡），每月提供 1000次免费的搜索额度，非常适合个人开发者进行测试和个人项目的初期开发。
使用load_dotenv
实现代码与配置分离
load_dotenv：环境变量.env文件

获取天气的api
可以免费获取天气的api，返回内容为json格式的数据（我的ME浏览器使用JSON Viewer美化了输出，按理说应该自带美化，不知道为什么没有）
https://wttr.in/{城市}?format=j1 例子：https://wttr.in/{北京}?format=j1
其余获取天气：openweather, 墨迹天气等
Tavily
tavily 搜索引擎API的用法
首先给大模型一个Prompt
根据天气推荐景点：
● 角色。智能旅行助手。任务是分析用户的请求，并使用可用工具一步一步地解决问题。
● 可用工具。查询指定城市天气：；根据城市和天气搜索推荐的旅游景点。
● 输出格式要求。你的每次回复必须遵守以下格式，包含一对Thought和Action...
● 重要提示（约束）。每次只输出一对Thought和Action，Action必须在同一行，不要换行，当收集到足够信息可以回答用户问题时，必须使用Action：Finish[最终答案] 格式结束
相关代码记录
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步一步地解决问题。

# 可用工具：
- 'get_weather(city: str)':查询指定城市的实时天气
- 'get_attraction(city: str, weather: str)':根据城市和天气搜索推荐的旅游景点

# 输出格式要求：
你的每次回复必须严格遵守以下格式，包含一对Thought和Action：

Thought：[你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具： function_name(arg_name="arg_value")
2. 结束任务： Finish[最终答案]

# 重要提示：
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""
# llm_model.py
from openai import OpenAI


class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务客户端。
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str)->str:
        """
        调用LLM API来生成回应
        :param prompt:
        :param sytem_prompt:
        :return:
        """
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误： {e}")
            return "Error:调用语言模型服务时出错。"

# smart_traveler.py
from dotenv import load_dotenv
from utils import get_weather, get_attraction
from llm_model import OpenAICompatibleClient
from Prompt import AGENT_SYSTEM_PROMPT
import os
import re

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_ID = os.getenv("MODEL_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction
}

llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# 初始化用户输入
user_prompt = "你好，请帮我查询一下今天上海的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求：{user_prompt}"]

print(f"用户输入：{user_prompt}\n" + "="*40)

# 运行主循环
for i in range(5):
    print(f"--- 循环 {i+1} ---\n")

    # 构建prompt
    full_prompt = "\n".join(prompt_history)

    # 调用LLM进行思考
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    # 模型可能会输出多余的Tought-Action，需要截断
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Tought-Action 对")
    print(f"模型输出：\n{llm_output}\n")
    prompt_history.append(llm_output)

    # 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "Error:未能解析到 Action字段。请确保你的回复严格遵循 'Thought: ... Action: ..'的格式。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print(f"任务完成，最终答案： {final_answer}")
        break
    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"Error:未定义的工具 '{tool_name}'"

    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*40)
    prompt_history.append(observation_str)
    
# utils.py
import requests
from tavily import TavilyClient


def get_weather(city: str) -> str:
    """
    通过调用wttr.in_API 查询真是天气
    :param city:
    :return:
    """

    # 请求API
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查相应状态码是否为200（成功）
        response.raise_for_status()
        # 解析返回的JSON数据
        data = response.json()
        # 提取当前天气状况
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']

        # 格式化成自然语言返回
        return f"{city}当前天气：{weather_desc}， 气温{temp_c}摄氏度"

    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误：查询天气时遇到网络问题 - {e}"

    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误：解析天气数据失败，可能是城市名称无效 - {e}"


def get_attraction(weather: str, city: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐
    :param weather:
    :param city:
    :return:
    """

    api_key = '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'
    if not api_key:
        return "Error:未配置TAVILY_API_KEY环境变量"

    # 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)

    # 构造精确查询
    query = f"'{city}'在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        # Tavily返回的结果已经非常干净，可以直接使用
        # response['answer'] 是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]

        # 如果没有综合性回答，则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐"

        return "根据搜索，为您找到以下信息：\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"Eror:执行Tavily搜索时出现问题 - {e}"

● Agent需要对大模型的输出做精细化的处理，包括边界值、异常场景等
Agent的三大范式
ReAct, Plan and Solve, Reflection
ReAct
Reasoning and acting，边想边做
工作模式：Thought->Action->Observation->Thought ...
待补充图以及数学公式
使用场景：
● 简单任务
● 获取外部知识
● 进行精确计算
● API交互相关
Plan and Solve
先规划再按计划执行
Reflection
ReAct范式
SerpAPI
SerpAPI 是一种搜索结果数据服务，它允许开发者通过 API 接口获取主流搜索引擎（如 Google、Bing、百度等）的搜索结果页面（SERP）数据。
https://serpapi.com
功能
● 能实现网页Google地搜索

● 谷歌财经、谷歌地图等大量功能都可以实现

API使用例子
params = {
    "engine": "google",
    "q": query,
    "api_key": api_key,
    "gl": "cn",  # 国家代码
    "hl": "zh-cn", # 语言代码
}
client = SerpApiClient(params)
ReAct实战经验总结
优势：高可解释性；动态规划和纠错能力；工具协同能力
局限：对LLM自身能力强依赖；执行效率不高（time, token）；可能陷入局部最优（走一步看一步，缺乏全局思维）
调试技巧：检查完整的提示词（例如qwen_3.7始终认为当前时间为2024年，通过在prompt说明时间是2026年修正结果）；分析原始输出和工具的输入与输出；调整提示词中的实例；尝试不同的模型或参数
代码
import json
import os
from serpapi import SerpApiClient
from typing import Dict, Any
from dotenv import load_dotenv


def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    :param query:
    :return:
    """
    print(f"🔍 正在执行 [SerpAPI] 网页搜索 {query}")
    try:
        load_dotenv()
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误： SERPAPI_API_KEY 不存在"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn", # 语言代码
        }

        client = SerpApiClient(params)
        results = client.get_dict()
        print("results: ", json.dumps(results, ensure_ascii=False, indent=4))

        # 智能解析：优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个organic_results的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的消息。"
    except Exception as e:
        return f"搜索时发生错误： {e}"


class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具
        :param name:
        :param description:
        :param func:
        :return:
        """
        if name in self.tools:
            print(f"警告：工具 '{name}' 已存在，将被覆盖。")

        self.tools[name] = {"description": description, "func": func}
        print(f"工具 {name} 已完成注册")

    def getTool(self, name: str) -> callable:
        """
        根据名称获取指定工具的执行函数
        :param name:
        :return:
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串
        :return:
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])


# 工具初始化与使用示例
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册实战搜索工具
    search_descrption = "一个网页搜索引擎。当你需要回答关于时事，事实以及在你的知识库中找不到的信息时，应当使用此工具。"
    toolExecutor.registerTool("Search", search_descrption, search)

    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 智能体的Action调用，这次问一个实时性的问题
    tool_name = "Search"
    tool_input = "介绍一下openclaw"
    print(f"\n--- 执行 Action: Search[{tool_input}] ---")

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误： 未找到名为 '{tool_name} 的工具。'")




import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any, List

# 加载环境变量
load_dotenv()


class LLMProxy:
    """
    用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str=None, apiKey: str=None, baseUrl:str=None, timeout: int=None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载
        :param model:
        :param apiKey:
        :param baseUrl:
        :param timeout:
        """
        self.model = model or os.getenv("MODEL_ID")
        apiKey = apiKey or os.getenv("API_KEY")
        baseUrl = baseUrl or os.getenv("BASE_URL")
        timeout = timeout or int(os.getenv("TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须要被提供或在.env文件中定义")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float=0) -> str:
        """
        调用大语言模型进行综合和思考，并返回其响应
        :param messages:
        :param temperature:
        :return:
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            # 处理流式响应
            print(" 大语言模型响应成功")
            collected_content = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f" 调用LLM API时发送错误： {e}")
            return None


# 客户端使用示例
if __name__ == '__main__':
    try:
        llmClient = LLMProxy()

        exampleMessages = [
            {'role': 'system', 'content': "You are a helpful assistant that writes Python code"},
            {'role': 'user', 'content': "写一个冒泡排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
from llm_client import LLMProxy
from utils import search, ToolExecutor
from REACT_PROMPT_TEMPLATE import REACT_PROMPT_TEMPLATE
import re


class ReActAgent:
    def __init__(self, llm_client: LLMProxy, tool_exexcutor: ToolExecutor, max_steps: int=5):
        self.llm_client = llm_client
        self.tool_executor = tool_exexcutor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(tools=tools_desc, question=question, history=history_str)

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("错误： LLM未能返回有效响应")
                break

            thought, action = self._parse_output(response_text)
            if thought:
                print(f" 思考： {thought}")
            if not action:
                print("警告：未能解析出有效的Action，流程终止。")
                break
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = self._parse_action_input(action)
                print(f" 最终答案： {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                self.history.append("Observation: 无效的Action格式， 请检查。")
                continue

            print(f" 行动： {tool_name}[{tool_input}]")
            tool_function = self.tool_executor.getTool(tool_name)
            observation = tool_function(tool_input) if tool_function else f"错误：未找到命名为 '{tool_name}' 的工具。"

            print(f" 观察: {observation}")
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        print("已达最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        return (match.group(1), match.group(2)) if match else (None, None)

    def _parse_action_input(self, action_text: str):
        match = re.match(r"\w+\[(.*)\]", action_text, re.DOTALL)
        return match.group(1) if match else ""


if __name__ == '__main__':
    llm = LLMProxy()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_exexcutor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)

REACT_PROMPT_TEMPLATE = """
你是一个有能力调用外部工具的智能助手。

# 可用工具：
{tools}

# 输出格式要求：
你的每次回复必须严格遵守以下格式，包含一对Thought和Action：

Thought：[你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具： {{tool_name}}[{{tool_input}}]
2. 结束任务： Finish[最终答案]

# 重要提示：
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束
- 当前时间为2026年6月28日

现在，请开始解决以下问题
Question: {question}
History: {history}
"""

Plan and Solve范式
规划阶段--->执行阶段
待补充流程图
Plan-and-Solve经验总结
优势：任务可靠性更高（结构化思考，进度追踪，失败恢复）；复杂任务分解能力（DAG/DCA）；更好的可观测性（debug）；资源效率（按需加载，避免浪费）
劣势：规划准确性依赖LLM；状态管理复杂度
适用场景：多步数学应用题；需要整合多个信息源的报告撰写；代码生成
Plan and Solve实战
● 两个prompt：planner（规划）和exeutor（执行）
● python内置的eval和ast.literal_eval的区别：两者都能解析基本数据类型（比如元组、列表等），但前者还可以计算表达式（计算、修改文件等），后者不能执行表达式更加安全
● PlanAndSolveAgent
正则表达式的基本用法需要学习
代码
import ast
from llm_client import LLMProxy

load_dotenv()

# 规划器定义
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个或多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划，```python与```作为前后缀是必要的：
```python
["步骤1","步骤2","步骤3", ...]
```
"""


class Planner:
    def __init__(self, llm_client: LLMProxy):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        response_text = self.llm_client.think(messages=messages) or ""
        print(f" 计划已生成:\n{response_text}")

        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f" 解析计划时出错: {e}")
            print(f" 原始响应: {response_text}")
            return []
        except Exception as e:
            print(f" 解析计划时发生未知错误: {e}")
            return []
from llm_client import LLMProxy

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步一步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话

# 原始问题:
{question}

# 完整计划：
{plan}

# 历史步骤与结果
{history}

# 当前步骤
{current_step}

请仅输出针对“当前步骤”地回答：
"""


class Executor:
    def __init__(self, llm_client: LLMProxy):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        history = ""
        final_answer = ""

        print("--- 正在执行计划 ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(question=question, plan=plan, history=history, current_step=step)
            messages = [{"role": "user", "content": prompt}]

            response_text = self.llm_client.think(messages=messages) or ""
            history += f"步骤 {i}: {step}\n结果：{response_text}\n\n"
            final_answer = response_text
            print(f" 步骤 {i} 已完成，结果：{final_answer}")

        return final_answer

from llm_client import LLMProxy
from planner import Planner
from executor import Executor


class PlanAndSolveAgent:
    def __init__(self, llm_client: LLMProxy):
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str):
        print(f"\n--- 开始处理问题 ---\n问题：{question}")
        plan = self.planner.plan(question)
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return
        final_answer = self.executor.execute(question, plan)
        print(f"\n--- 任务完成 ---\n最终答案：{final_answer}")


if __name__ == '__main__':
    try:
        llm_client = LLMProxy()
        agent = PlanAndSolveAgent(llm_client)
        question = "一个水果店周一卖出了15个苹果，周二卖出的苹果数量是周一的两倍，周三卖出的数量比周二少了5个。请问这三天一共卖出了多少个苹果？"
        agent.run(question=question)
    except ValueError as e:
        print(e)

Reflection范式
执行execution->反思->优化
执行生成一个初始解（React/PlanAndSolve），通过prompt+llm进行反思（Review角色，可以使用不同的LLM；需要短期记忆/长期记忆），优化（将反思内容+query再次送入大模型）
画图及公式
Reflection实战经验总结
主要成本：
● 模型调用开销增加（反思+优化）
● 任务延迟显著调高（串行执行，先反思再执行，不适合对实时要求高的需求）
● 提示工程复杂度上升（例如模型第一次输出就被视为最优，没有继续优化）
核心收益：
● 解决方案质量的跃迁（以成本换质量）
● 鲁棒性与可靠性增强
使用场景：
● 生成关键的业务代码或技术报告
● 在科学研究中进行复杂的逻辑推演
● 需要深度分析和规划的决策系统
代码
from typing import List, Dict, Any
from llm_client import LLMProxy


# 记忆模块
class Memory:
    """
    一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹。
    """
    def __init__(self):
        # 初始化一个空列表来存储所有记录
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        向记忆中添加一条新纪录。
        :param record_type: 记录类型（execution 或 reflection）
        :param content: 记录的具体内容（例如，生成的代码或反思的反馈）
        :return:
        """
        self.records.append({"type": record_type, "content": content})
        print(f" 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """
        将所有记忆记录格式化为一个连贯的字符串文本，用于构建提示词
        :return:
        """
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"---上一轮尝试（代码）---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"---评审员反馈---\n{record['content']}\n\n"
        return trajectory.strip()  # strip() 去除字符串首尾处的空白字符（包含空格 \t \n等）

    def get_last_execution(self) -> str:
        """
        获取最近一次的执行结果（例如，最新生成的代码）。
        :return:
        """
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
            return None
from typing import List, Dict, Any
from llm_client import LLMProxy
from memory import Memory

# 初始提示词
INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求：{task}

请直接输出代码，不要包含任何额外的解释。
"""

# 反思提示词
REFLECTION_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Pythondaima，并专注于找出在**算法效率**上的主要瓶颈。

# 原始任务：
{task}

# 待审查的代码：
```python
{code}
```

请分析改代码的时间复杂度，并思考是否存在一种**算法上更优**的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛选法代替试除法）。
如果代码在算法层面已经达到最优，才能回答无需改进。

请直接输出你的反馈，不要包含任何额外的解释。
"""

# 优化提示词
REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务：
{task}

# 你上一轮尝试的代码：
{last_code_attempt}

# 评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

请直接输出优化后的代码，不要包含任何额外的解释。
"""


class ReflectionAgent:
    """
    一个Reflection范式的智能体，遵循 执行——>反思——>优化 的逻辑
    """
    def __init__(self, llm_client: LLMProxy, max_interation: int=3):
        """
        初始化Reflection范式的智能体
        :param llm_client:
        :param max_interation:
        """
        self.llm_client = llm_client
        self.memory = Memory()
        self.iteration = max_interation

    def _get_llm_response(self, prompt: str) -> str:
        """一个辅助方法，用于调用LLM并获取完整的流式响应"""
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(messages=messages) or ""
        return response_text

    def run(self, task: str):
        print(f"\n--- 开始处理任务 ---任务：{task}")

        # 初始执行
        print(f"\n--- 正在进行初始尝试 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        # 迭代循环，反思优化
        for i in range(self.iteration):
            print(f"\n--- 第 {i+1}/{self.iteration} 轮迭代 ---")

            # 反思
            print(f"\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflection_prompt = REFLECTION_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self._get_llm_response(reflection_prompt)
            self.memory.add_record("reflection", feedback)

            # 检查是否需要停止
            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                print(f"\n 反思认为代码已无需改进，任务完成")
                break

            # 优化
            print("\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)

        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码：\n{final_code}")
        return final_code


if __name__ == '__main__':
    # 初始化LLM客户端
    try:
        llm_client = LLMProxy()
    except Exception as e:
        print(f"初始化LLM客户端时出错：{e}")
        exit()

    # 初始化Reflection智能体，设置最多迭代2轮
    agent = ReflectionAgent(llm_client=llm_client, max_interation=2)

    # 定义任务并运行智能体
    task = "编写一个Python函数，找出1到n之间所有的素数（prime numbers）"
    agent.run(task)

# 使用了qwen3-coder-plus模型，其余与plan_and_solve相同
面试题
除了ReAct范式，Agent还有什么新范式或者架构？
ReAct: Reasoning + Acting（效率低；缺乏全局规划）
● LATS
LATS 全称是 Language Agent Tree Search（语言智能体树搜索），是由普林斯顿大学等机构的研究人员提出的一种先进的大语言模型（LLM）推理框架。
它的核心思想是将大语言模型作为智能体（Agent），并结合蒙特卡洛树搜索（MCTS）算法，让模型在解决复杂问题时能够进行多步规划、自我反思和纠错，从而大幅提升推理的准确性
传统的 LLM 生成通常是单向的（自回归），一旦中间某一步出错，后续很容易“一错到底”。而 LATS 引入了树搜索机制，其工作流包含以下关键步骤：
1. 状态评估（Value Function）：LLM 不仅生成下一步动作，还会对当前的状态（或生成的中间结果）进行打分评估，判断这步走得好不好。
2. 环境反馈（Environment Feedback）：LATS 可以接入外部环境（如代码解释器、搜索引擎、数学计算器等）。如果生成了代码，环境会运行它并返回报错或结果；如果生成了搜索词，环境会返回网页内容。
3. 自我反思与纠错（Reflection）：当环境返回错误（如代码报错）或评估分数较低时，LLM 会将这些“负面反馈”作为上下文，重新思考并生成新的解决方案。
4. 树搜索（MCTS）：系统会将多种可能的解决路径构建成一棵树，通过模拟、扩展和回溯，不断寻找得分最高的最优路径，最终输出最佳答案。

● RAISE
RAISE Agent 通常指的是一种基于 ReAct 框架增强的高级 AI 智能体架构。它的核心创新在于引入了模拟人类短期和长期记忆的双组件记忆系统，旨在解决传统大语言模型在长对话中容易丢失上下文的问题。
RAISE 的全称通常与“通过草稿和示例进行推理和行动”（Reasoning and Acting with Scratchpad and Examples）相关。其核心由两个记忆组件构成：
1. 短期记忆（Scratchpad / 便签簿）：用于捕捉和处理最近交互的关键信息和结论，相当于人类的工作记忆。
2. 长期记忆（Example Pool / 示例库）：这是一个包含历史类似示例的数据集，用于检索与当前对话上下文相关的信息。
通过这种双组件机制，RAISE 能够显著提升智能体在复杂多轮对话中的连贯性和上下文意识，使其在效率和输出质量上优于基础的 ReAct 方法。
局限：
● 复杂逻辑推理困难：由于严重依赖记忆中的“类似示例”，当遇到需要多步推理、条件判断或抽象逻辑的任务时，RAISE 容易迷失在记忆片段中，难以像人类那样进行深度推理。
● 容易产生幻觉：如果长期记忆库中没有足够准确或相关的例子，智能体可能会“编造”内容来填补空白。此外，如果角色定义不明确，它可能会表现出与当前任务无关的能力（例如，一个销售 Agent 突然开始编写 Python 代码），从而提供误导性信息。
架构选择
任务明确，例如文档生成，解决数学题——Plan and solve
开放性问题，需要探索和调整，例如问答交互系统——ReAct
对推理质量高，可以忍受高成本——LATS
还可以结合多种架构构造Agent：例如plan and solve做整体框架，某些需要灵活处理的问题采用React，某些需要高质量输出的采用LATS或者加入reflection。现实问题靠单一架构难以解决，混合模式非常常见
