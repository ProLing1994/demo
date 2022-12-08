from pinyin import get_pinyin


def dummy_test():
    assert get_pinyin('你好', 'ni3 hao3')

if __name__ == "__main__":
    print(" ".join(get_pinyin("你好？中文！中文的，符号")))
    print(" ".join(get_pinyin("持起红缨枪追赶对方半公里")))
    print(" ".join(get_pinyin("过把瘾的演员有什么")))
    print(" ".join(get_pinyin("放一首情歌")))
    print(" ".join(get_pinyin("校园青春校园小说有什么")))
    print(" ".join(get_pinyin("大智慧阿思达克通讯社")))
    print(" ".join(get_pinyin("今天星期五，好开心啊")))
    print(" ".join(get_pinyin("科技构筑美好交通未来")))
