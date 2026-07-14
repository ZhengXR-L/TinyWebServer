为实现这一目标，需要创建三样东西：
● 助手节点：装载了LLM，用于判断是否需要调用工具
● 工具节点：一个可获取城市天气的工具。这里被我们简化，对任何城市都输出晴天
● 条件边：连接助手节点与工具节点。根据助手节点的输出，决定是否调用工具
有了上述三样东西后，即有了构建状态图的基础原料。但要让状态图运行起来，还需要定义节点和边之间的关系。首先需要将节点添加到StateGraph实例中，然后以正确的顺序桥接它们。
有向图：
● 有向无环图DAG
● 有向循环图DCG
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools import tool

# 关键节点: ToolNode
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig


load_dotenv()

# 助手节点
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_ID"),
    temperature=0.7
)


# 工具函数
@tool
def get_weather(city: str) -> str:
    '''
    根据给定city获取对应的天气
    '''
    return f"查询结果: {city}天气晴朗，万里碧空飘着朵朵白云"


# 工具节点
tools = [get_weather]
tool_node = ToolNode(tools)


# 创建带工具的LLM助手节点
def assistant(state: MessagesState, config: RunnableConfig):
    system_prompt = 'You are a helpful assistant that can check weather.'
    all_messages = [SystemMessage(system_prompt)] + state['messages']
    model = llm.bind_tools(tools)
    return {'messages': [model.invoke(all_messages)]}


# 创建条件边
def should_continue(state: MessagesState, config: RunnableConfig):
    messages = state['messages']
    last_message = messages[-1]
    # 如果最后一条消息是LLM决定使用工具调用，则继续
    if last_message.tool_calls:
        return 'continue'
    # 否则结束
    return 'end'


# 创建图
builder = StateGraph(MessagesState)

# 添加节点
builder.add_node('assistant', assistant)
builder.add_node('tool', tool_node)

# 添加边（从START到assistant）起点必须是START，终点必须是END
builder.add_edge(START, 'assistant')

# 添加条件边（根据assistant的输出，决定是否调用工具）
builder.add_conditional_edges(
    # 条件边连接的节点
    'assistant',
    # 条件边判断逻辑
    should_continue,
    # 条件边判断逻辑为True时，调用tool节点
    {
        'continue': 'tool',
        'end': END,
    },
)

# 添加边：调用工具节点后回到assistant
builder.add_edge('tool', 'assistant')

# 编译图
my_graph = builder.compile(name='my-graph')
my_graph

# 调用图
response = my_graph.invoke({"messages": [HumanMessage(content="重庆天气怎么样？")]})
for message in response['messages']:
    message.pretty_print()
