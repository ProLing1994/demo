class XmlApi():
    """
    XmlApi
    """

    def __init__(self, xml_path, frame_num):
        self.xml_path = xml_path
        self.frame_num = frame_num

        self.analysis_xml()

    def analysis_xml(self):
        with open(self.xml_path, "r", encoding='gbk') as f:

            self.frame_list = []
            for line in f:
                if "<frame>" in line:
                    self.frame_dict = {}
                if "<frameindex>" in line:
                    self.frame_dict['frameindex'] = line.split('<frameindex>')[1].split('</frameindex>')[0]
                if "<pts>" in line:
                    self.frame_dict['pts'] = line.split('<pts>')[1].split('</pts>')[0]
                if "</frame>" in line:
                    self.frame_list.append(self.frame_dict)
        
        self.frame_rate = int( float(self.frame_list[-1]['frameindex']) / self.frame_num + 0.5 )
        print(self.frame_list[-1]['frameindex'], self.frame_num, self.frame_rate)

    def find_pts(self, frame_idx):
        find_pts = 0 
        find_frame_idx = frame_idx * self.frame_rate

        for idx in range(len(self.frame_list)):
            if int(self.frame_list[idx]['frameindex']) < find_frame_idx:
                find_pts = int(self.frame_list[idx]['pts'])
            elif int(self.frame_list[idx]['frameindex']) >= find_frame_idx:
                find_pts = int(self.frame_list[idx]['pts'])
                break
        
        return find_pts
