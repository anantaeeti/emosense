import json, numpy as np, torch, streamlit as st
from PIL import Image, ImageOps
from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                          AutoTokenizer, AutoModelForSequenceClassification)
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from models import (TextLSTM, FaceCNN, encode, face_eval_tf,
                    TEXT_LABELS, FACE_LABELS)

st.set_page_config(page_title="EmoSense", page_icon="🙂")
device = "cpu"
VIT_ID = "trpakov/vit-face-expression"
TEXT2_ID = "j-hartmann/emotion-english-distilroberta-base"
GOEMO_ID = "SamLowe/roberta-base-go_emotions"
HSE_ORDER = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# one shared label list for the text side
TEXT_ALL = ["anger", "disgust", "fear", "joy", "love", "neutral", "sadness", "surprise"]

# how the 28 GoEmotions labels are grouped into the shared text labels
GO_GROUPS = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "optimism",
            "relief", "pride", "desire"],
    "love": ["love", "caring", "admiration"],
    "neutral": ["neutral"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
}

# map text labels into the 7 face classes so text and face can be fused
TEXT_TO_COMMON = {"anger": "angry", "disgust": "disgust", "fear": "fear", "joy": "happy",
                  "love": "happy", "neutral": "neutral", "sadness": "sad", "surprise": "surprise"}

@st.cache_resource
def load_models():
    vocab = json.load(open("saved/vocab.json"))
    cfg = json.load(open("saved/text_config.json"))
    tm = TextLSTM(cfg["vocab_size"])
    tm.load_state_dict(torch.load("saved/text_lstm.pth", map_location=device))
    tm.eval()
    fm = FaceCNN()
    fm.load_state_dict(torch.load("saved/face_cnn.pth", map_location=device))
    fm.eval()
    proc = AutoImageProcessor.from_pretrained(VIT_ID)
    vit = AutoModelForImageClassification.from_pretrained(VIT_ID).eval()
    order = [FACE_LABELS.index(vit.config.id2label[i].lower())
             for i in range(len(vit.config.id2label))]
    hse = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")
    tok = AutoTokenizer.from_pretrained(TEXT2_ID)
    rob = AutoModelForSequenceClassification.from_pretrained(TEXT2_ID).eval()
    go_tok = AutoTokenizer.from_pretrained(GOEMO_ID)
    go = AutoModelForSequenceClassification.from_pretrained(GOEMO_ID).eval()
    return tm, fm, vocab, cfg["max_len"], proc, vit, order, hse, tok, rob, go_tok, go

(text_model, face_model, vocab, max_len,
 vit_proc, vit_model, vit_order, hse_model,
 rob_tok, rob_model, go_tok, go_model) = load_models()

st.sidebar.header("Face ensemble weights")
w_cnn = st.sidebar.slider("Your CNN", 0.0, 1.0, 0.2)
w_vit = st.sidebar.slider("Pretrained ViT (FER-2013)", 0.0, 1.0, 0.4)
w_hse = st.sidebar.slider("HSEmotion (AffectNet)", 0.0, 1.0, 0.4)
st.sidebar.header("Text ensemble weights")
w_lstm = st.sidebar.slider("Your LSTM", 0.0, 1.0, 0.1)
w_rob = st.sidebar.slider("DistilRoBERTa (6 datasets)", 0.0, 1.0, 0.45)
w_go = st.sidebar.slider("RoBERTa (GoEmotions)", 0.0, 1.0, 0.45)
st.sidebar.header("Display")
min_conf = st.sidebar.slider("Show every emotion above", 0.05, 0.50, 0.15)

def predict_text(text):
    ids = torch.tensor([encode(text, vocab, max_len)])
    with torch.no_grad():
        p6 = torch.softmax(text_model(ids, torch.tensor([ids.shape[1]])), 1)[0].numpy()
        enc = rob_tok(text, return_tensors="pt", truncation=True, max_length=128)
        pr = torch.softmax(rob_model(**enc).logits, 1)[0].numpy()
        enc2 = go_tok(text, return_tensors="pt", truncation=True, max_length=128)
        pg = torch.sigmoid(go_model(**enc2).logits)[0].numpy()   # multi-label: sigmoid

    p_lstm = np.zeros(len(TEXT_ALL))
    for lab, p in zip(TEXT_LABELS, p6):
        p_lstm[TEXT_ALL.index(lab)] += p

    p_rob = np.zeros(len(TEXT_ALL))
    for i, p in enumerate(pr):
        p_rob[TEXT_ALL.index(rob_model.config.id2label[i].lower())] = p

    name2p = {go_model.config.id2label[i].lower(): float(p) for i, p in enumerate(pg)}
    p_go = np.zeros(len(TEXT_ALL))
    for grp, names in GO_GROUPS.items():
        p_go[TEXT_ALL.index(grp)] = max(name2p.get(n, 0.0) for n in names)
    p_go = p_go / p_go.sum() if p_go.sum() > 0 else p_go

    fused = w_lstm * p_lstm + w_rob * p_rob + w_go * p_go
    fused = fused / fused.sum() if fused.sum() > 0 else p_rob
    return fused, p_lstm, p_rob, p_go

def crop_face(img):
    try:
        import cv2
        names = ["haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_default.xml"]
        cascades = [cv2.CascadeClassifier(cv2.data.haarcascades + n) for n in names]
        profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

        def detect(pil_img, casc_list):
            gray = cv2.equalizeHist(np.array(pil_img.convert("L")))
            for c in casc_list:
                faces = c.detectMultiScale(gray, 1.05, 4, minSize=(30, 30))
                if len(faces):
                    return max(faces, key=lambda f: f[2] * f[3])
            return None

        def crop_box(pil_img, box):
            x, y, w, h = box
            pad = int(0.1 * w)
            return pil_img.crop((max(0, x - pad), max(0, y - pad),
                                 min(pil_img.width, x + w + pad), min(pil_img.height, y + h + pad)))

        for ang in [0, 15, -15, 30, -30]:
            rot = img.rotate(ang, resample=Image.BILINEAR)
            box = detect(rot, cascades)
            if box is not None:
                return crop_box(rot, box), True

        for flip in [False, True]:
            im = img.transpose(Image.FLIP_LEFT_RIGHT) if flip else img
            box = detect(im, [profile])
            if box is not None:
                return crop_box(im, box), True
    except Exception as e:
        st.warning(f"Face detector error: {e}")
    W, H = img.size
    s = min(W, H)
    return img.crop(((W - s) // 2, (H - s) // 2, (W + s) // 2, (H + s) // 2)), False

def predict_face(img):
    face, found = crop_face(img)
    face_rgb = face.convert("RGB")
    x = face_eval_tf(face).unsqueeze(0)
    with torch.no_grad():
        p1 = torch.softmax(face_model(x), 1)
        p2 = torch.softmax(face_model(torch.flip(x, [3])), 1)
        p_cnn = ((p1 + p2) / 2)[0].numpy()
        v1 = torch.softmax(vit_model(**vit_proc(images=face_rgb, return_tensors="pt")).logits, 1)
        v2 = torch.softmax(vit_model(**vit_proc(images=ImageOps.mirror(face_rgb), return_tensors="pt")).logits, 1)
        pv = ((v1 + v2) / 2)[0].numpy()
    p_vit = np.zeros(len(FACE_LABELS))
    for j, k in enumerate(vit_order):
        p_vit[k] = pv[j]

    _, sc = hse_model.predict_emotions(np.array(face_rgb), logits=False)
    sc = np.asarray(sc).ravel()
    p_hse = np.zeros(len(FACE_LABELS))
    for name, s in zip(HSE_ORDER, sc):
        if name in FACE_LABELS:           # contempt has no match, so it is dropped
            p_hse[FACE_LABELS.index(name)] += s
    p_hse = p_hse / p_hse.sum() if p_hse.sum() > 0 else p_hse

    fused = w_cnn * p_cnn + w_vit * p_vit + w_hse * p_hse
    # optional: models over-predict neutral; remove the # on the next line to shrink it a bit
    # fused[FACE_LABELS.index("neutral")] *= 0.75
    fused = fused / fused.sum() if fused.sum() > 0 else p_vit
    return fused, p_cnn, p_vit, p_hse, face, found

def top3(p, labels):
    o = np.argsort(p)[::-1][:3]
    return "  ·  ".join(f"{labels[i]} {p[i]*100:.0f}%" for i in o)

def top(p, labels=FACE_LABELS):
    return f"{labels[p.argmax()]} ({p.max()*100:.0f}%)"

def show(probs, labels):
    order = np.argsort(probs)[::-1]
    # the top emotion is always shown; others are added if they pass the threshold (max 3)
    picks = [order[0]] + [i for i in order[1:] if probs[i] >= min_conf][:2]
    txt = "  +  ".join(f"{labels[i]} ({probs[i]*100:.0f}%)" for i in picks)
    if len(picks) == 1:
        st.success(f"Prediction: {txt}")
    else:
        st.success(f"Mixed emotions: {txt}")
    st.caption("Top 3:  " + top3(probs, labels))
    st.bar_chart({labels[i]: float(probs[i]) for i in order})

st.title("EmoSense: multimodal emotion detection")
tab1, tab2, tab3 = st.tabs(["Text", "Face", "Combined"])

with tab1:
    t = st.text_area("Type something", key="t1")
    if st.button("Analyze text") and t.strip():
        fused_t, p_lstm, p_rob, p_go = predict_text(t)
        show(fused_t, TEXT_ALL)
        st.caption(f"Your LSTM: {top(p_lstm, TEXT_ALL)}  |  DistilRoBERTa: {top(p_rob, TEXT_ALL)}  |  "
                   f"GoEmotions: {top(p_go, TEXT_ALL)}")

with tab2:
    src = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
    f = st.file_uploader("Face image", type=["jpg", "jpeg", "png"]) if src == "Upload" \
        else st.camera_input("Take a photo")
    if f:
        fused, p_cnn, p_vit, p_hse, face, found = predict_face(Image.open(f).convert("RGB"))
        st.image(face.convert("L").resize((48, 48)), width=150,
                 caption="face detected" if found else "no face found, centre crop used")
        if not found:
            st.warning("No face detected, so this result is unreliable. Face the camera straight on, "
                       "keep your whole face in the frame, and retake.")
        show(fused, FACE_LABELS)
        st.caption(f"Your CNN: {top(p_cnn)}  |  ViT: {top(p_vit)}  |  HSEmotion: {top(p_hse)}")

with tab3:
    t2 = st.text_area("Text", key="t2")
    f2 = st.file_uploader("Face image", type=["jpg", "jpeg", "png"], key="f2")
    w = st.slider("Weight on text vs face", 0.0, 1.0, 0.5)
    if st.button("Fuse") and t2.strip() and f2:
        tp, _, _, _ = predict_text(t2)
        fp, _, _, _, face2, found2 = predict_face(Image.open(f2).convert("RGB"))
        st.image(face2.convert("L").resize((48, 48)), width=150,
                 caption="face detected" if found2 else "no face found, centre crop used")
        text_common = np.zeros(len(FACE_LABELS))
        for lab, pr in zip(TEXT_ALL, tp):
            text_common[FACE_LABELS.index(TEXT_TO_COMMON[lab])] += pr
        fused = w * text_common + (1 - w) * fp
        fused /= fused.sum()
        show(fused, FACE_LABELS)
        st.caption(f"Text alone: {top3(text_common, FACE_LABELS)}")
        st.caption(f"Face alone: {top3(fp, FACE_LABELS)}")