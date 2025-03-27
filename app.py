import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import joblib
import os


# 1. 数据加载（自动处理编码和列名）
@st.cache_data
def load_data():
    # 尝试多种编码方式
    encodings = ['utf-8', 'gbk', 'utf-16', 'utf-8-sig']

    for encoding in encodings:
        try:
            df = pd.read_csv('crop_data.csv', encoding=encoding)
            # 统一处理列名（去除首尾空格和特殊字符）
            df.columns = df.columns.str.strip().str.replace(' ', '').str.replace('(', '').str.replace(')', '')
            return df
        except (UnicodeDecodeError, FileNotFoundError) as e:
            continue

    st.error("无法加载数据文件，请检查：\n1. 文件是否存在\n2. 文件编码格式")
    st.stop()


# 2. 模型训练
def train_model(df):
    # 列名映射（处理各种可能的列名变体）
    col_mapping = {
        'month': ['种植月', '月份', 'month'],
        'temp': ['温度℃', '温度', 'temp'],
        'rain': ['降雨mm', '降雨', 'rain'],
        'ph': ['土壤pH', 'pH值', 'ph'],
        'crop': ['作物', 'crop']
    }

    # 自动匹配列名
    selected_cols = {}
    for col_type, possible_names in col_mapping.items():
        for name in possible_names:
            if name in df.columns:
                selected_cols[col_type] = name
                break
        else:
            st.error(f"找不到匹配的列：{possible_names}")
            st.stop()

    X = df[[selected_cols['month'], selected_cols['temp'], selected_cols['rain'], selected_cols['ph']]]
    y = df[selected_cols['crop']]

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)

    joblib.dump(model, 'model.pkl')
    return model


# 3. 创建Streamlit界面
st.set_page_config(page_title="智能作物推荐系统", layout="wide")
st.title('🌱 智能作物推荐系统')

# 加载数据
try:
    df = load_data()
    st.success("数据加载成功！")
except Exception as e:
    st.error(f"数据加载失败：{str(e)}")
    st.stop()

# 输入面板
with st.sidebar:
    st.header("🛠️ 环境参数")
    month = st.slider("种植月份", 1, 12, 5)
    temp = st.slider("平均温度(℃)", 0, 40, 25)
    rain = st.number_input("降雨量(mm)", 0, 2000, 800)
    ph = st.slider("土壤pH值", 3.0, 9.0, 6.5)

# 主界面
if st.button('🌾 获取推荐'):
    try:
        model = joblib.load('model.pkl')
    except FileNotFoundError:
        with st.spinner('首次运行正在训练模型...'):
            model = train_model(df)

    try:
        # 准备输入数据（列顺序必须与训练时一致）
        input_data = pd.DataFrame([[month, temp, rain, ph]],
                                  columns=['种植月', '温度℃', '降雨mm', '土壤pH'])

        # 预测作物
        crop = model.predict(input_data)[0]
        st.success(f'## 推荐种植: {crop}')

        # 显示详细信息
        crop_info = df[df['作物'] == crop].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **📊 种植信息:**
            - 适宜温度: {crop_info['温度℃']}℃
            - 需水量: {crop_info['降雨mm']}mm
            - 最佳pH: {crop_info['土壤pH']}
            """)

        with col2:
            st.markdown(f"""
            **⚠️ 常见问题:**
            - {crop_info['常见问题']}
            - 预计产量: {crop_info['产量等级']}
            """)

    except Exception as e:
        st.error(f"预测失败: {str(e)}")

# 显示原始数据（调试用）
with st.expander("📁 查看原始数据"):
    st.dataframe(df)

# 添加免责声明
st.caption("注意：本推荐结果基于历史数据，实际种植请结合当地具体情况。")