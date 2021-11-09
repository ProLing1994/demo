from prosody import parse_cn_prosody_label_type1

if __name__ == "__main__":
    print(parse_cn_prosody_label_type1('妈妈#1当时#1表示#3，儿子#1开心得#2像花儿#1一样#4。', 'ma1 ma1 dang1 shi2 biao3 shi4 er2 zi5 kai1 xin1 de5 xiang4 huar1 yi2 yang4'))
    print(parse_cn_prosody_label_type1('遛弯儿#2都得#2躲远点#4。', 'liu4 wanr1 dou1 dei3 duo2 yuan2 dian3'))
    print(parse_cn_prosody_label_type1('油炸#1豆腐#1喷喷香#3，馓子#1麻花#2嘣嘣脆#3，姊妹#1团子#2数二姜#4。', 'you2 zha2 dou4 fu5 pen1 pen1 xiang1 san3 zi5 ma2 hua1 beng1 beng1 cui4 zi3 mei4 tuan2 zi5 shu3 er4 jiang1'))
