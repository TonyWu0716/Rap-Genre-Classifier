import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import io

# -------------------- 页面配置（必须放在最前面） --------------------
st.set_page_config(
    page_title="嘻哈风格识别 | 街头科幻",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------- 自定义 CSS（明亮街头涂鸦风格） --------------------
st.markdown("""
<style>
    /* 全局背景 - 明亮的街头涂鸦墙风格 */
    .stApp {
        background: linear-gradient(135deg, #f5e6d3 0%, #ffe6f0 100%);
        background-attachment: fixed;
    }
    /* 主容器 - 半透磨砂玻璃效果 */
    .main > div {
        background: rgba(255, 248, 240, 0.85);
        backdrop-filter: blur(8px);
        border-radius: 2rem;
        padding: 1.5rem;
        border: 2px solid #ff99cc;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1), 0 0 0 4px #ffccdd inset, 0 0 0 8px #ffffff80 inset;
    }
    /* 标题 - 街头喷漆风格 */
    h1 {
        background: linear-gradient(135deg, #ff66b2, #3399ff);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        text-align: center;
        font-family: 'Permanent Marker', 'Impact', cursive;
        font-size: 3rem;
        text-shadow: 4px 4px 0px #ffccdd;
        letter-spacing: 1px;
    }
    .sub {
        text-align: center;
        color: #cc3366;
        background: #ffffccaa;
        display: inline-block;
        width: auto;
        margin-bottom: 1rem;
        padding: 0.2rem 1rem;
        border-radius: 30px;
        font-family: monospace;
        font-weight: bold;
        border: 1px dashed #ff9966;
    }
    .stFileUploader > div {
        background: #fffaf2;
        border: 3px dashed #ff88bb;
        border-radius: 1.5rem;
        padding: 1rem;
        transition: 0.2s;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .stFileUploader > div:hover {
        border-color: #3399ff;
        background: #fff5e6;
        transform: scale(0.99);
    }
    .genre-bar {
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        background: #ffffffaa;
        padding: 0.3rem 0.8rem;
        border-radius: 50px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .genre-name {
        width: 120px;
        font-weight: bold;
        color: #cc3366;
        font-family: 'Courier New', monospace;
        text-transform: uppercase;
        font-size: 0.9rem;
        text-shadow: 1px 1px 0 #ffccdd;
    }
    .bar-bg {
        flex: 1;
        height: 28px;
        background: #ffe0e8;
        border-radius: 30px;
        overflow: hidden;
        box-shadow: inset 0 1px 3px #cc9999;
    }
    .bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff99cc, #66b5ff);
        border-radius: 30px;
        transition: width 0.4s ease-out;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 8px;
        color: #1a1a2e;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .percentage {
        min-width: 55px;
        text-align: right;
        color: #3399ff;
        font-family: monospace;
        font-weight: bold;
        background: #ffffffaa;
        padding: 0.2rem 0.5rem;
        border-radius: 20px;
    }
    footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.7rem;
        color: #cc6699;
        font-family: monospace;
        font-weight: bold;
    }
    .stButton button {
        background: #ffbbdd;
        color: #cc3366;
        border: none;
        border-radius: 40px;
        font-weight: bold;
        font-family: monospace;
    }
    .stButton button:hover {
        background: #ff99cc;
        color: white;
        transform: scale(1.02);
    }
    body::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        background-image: radial-gradient(circle at 10% 20%, rgba(255,200,230,0.3) 2%, transparent 2.5%),
                          radial-gradient(circle at 80% 70%, rgba(100,150,255,0.2) 1.5%, transparent 2%);
        background-size: 55px 55px, 80px 80px;
        z-index: 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- 配置参数（请根据您的模型修改） --------------------
class Config:
    model_path = 'best_model_mel.pth'      # 模型文件路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_rate = 22050
    duration = 5.0
    n_mels = 128
    hop_length = 512
    n_fft = 2048
    img_height = 128
    img_width = 500
    num_classes = 8
    # ⚠️ 请替换为您真实的7种风格名称，顺序必须与训练时严格一致
    class_names = [
        'Boombap', 'Drill', 'Glo', 'Memphis',
        'Pop_rap', 'Rage', 'Regalia','Trap'
    ]

cfg = Config()

# -------------------- 模型定义 --------------------
class CNNLSTMAttention(nn.Module):
    def __init__(self, input_channels=1, cnn_out_channels=64, lstm_hidden_size=128,
                 lstm_num_layers=2, dropout=0.3, num_classes=7):
        super(CNNLSTMAttention, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(),
        )
        self.freq_fixed = 8
        self.adaptive_freq = nn.AdaptiveAvgPool2d((self.freq_fixed, None))
        self.lstm_input_dim = cnn_out_channels * self.freq_fixed
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size // 2, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        cnn_feat = self.adaptive_freq(cnn_feat)
        b, c, f, t = cnn_feat.shape
        cnn_feat = cnn_feat.permute(0, 3, 1, 2).contiguous()
        cnn_feat = cnn_feat.view(b, t, c * f)
        lstm_out, _ = self.lstm(cnn_feat)
        att_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(att_weights * lstm_out, dim=1)
        logits = self.classifier(context)
        return logits

# -------------------- 加载模型 --------------------
@st.cache_resource
def load_model():
    model = CNNLSTMAttention(num_classes=cfg.num_classes).to(cfg.device)
    model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))
    model.eval()
    return model

model = load_model()
st.success("🎤 模型加载成功，等待音频上传...")

# -------------------- 音频预处理函数（接收 bytes） --------------------
def preprocess_audio_from_bytes(audio_bytes):
    """从字节流加载音频，返回模型输入张量 (1,1,128,500)"""
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=cfg.sample_rate, duration=cfg.duration)
    target_len = int(cfg.duration * cfg.sample_rate)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.sample_rate, n_mels=cfg.n_mels,
        n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    spec = torch.from_numpy(mel_db).float().unsqueeze(0).unsqueeze(0)
    spec = torch.nn.functional.interpolate(
        spec, size=(cfg.img_height, cfg.img_width),
        mode='bilinear', align_corners=False
    )
    return spec.to(cfg.device)

# -------------------- UI 布局 --------------------
st.markdown("<h1>🎧 STREET WAVE</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center'><div class='sub'>［ AI 嘻哈风格鉴定 | 霓虹熔炉 ］</div></div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "点击或拖拽上传 MP3 / WAV",
    type=['mp3', 'wav'],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # 读取音频字节数据（用于播放和预处理）
    audio_bytes = uploaded_file.read()
    
    # 显示音频播放器（增加“播放键”）
    st.markdown("**🎵 试听片段**")
    st.audio(audio_bytes, format=uploaded_file.type)
    
    with st.spinner("🔮 正在分析音频... 旋律抽取中"):
        try:
            input_tensor = preprocess_audio_from_bytes(audio_bytes)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            results = {cfg.class_names[i]: float(probs[i]) * 100 for i in range(cfg.num_classes)}
            sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)

            st.markdown("### 🔥 风格成分分析")
            for genre, percent in sorted_items:
                percent_fixed = f"{percent:.1f}%"
                bar_html = f"""
                <div class="genre-bar">
                    <div class="genre-name">{genre}</div>
                    <div class="bar-bg">
                        <div class="bar-fill" style="width: {percent}%;">{int(percent)}%</div>
                    </div>
                    <div class="percentage">{percent_fixed}</div>
                </div>
                """
                st.markdown(bar_html, unsafe_allow_html=True)

            with st.expander("📊 查看详细概率分布"):
                st.bar_chart(results)

            st.balloons()
            st.markdown("<footer>⚡ CNN + LSTM + Attention · 实时推理 ⚡</footer>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"处理出错: {str(e)}\n请检查文件格式或联系开发者")
else:
    st.info("✨ 上传一首嘻哈 beats 或说唱，AI 将为你分析风格成分。")