from collections import Counter
def fuse_emotions(text_emotion=None, speech_emotion=None, video_emotion=None):
    items = [e for e in (text_emotion, speech_emotion, video_emotion) if e]
    if not items: return None
    cnt = Counter(items)
    return cnt.most_common(1)[0][0]
