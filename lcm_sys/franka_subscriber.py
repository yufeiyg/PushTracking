import lcm
from lcm_sys.lcm_types.lcm_rb_pose.lcmt_sampling_c3_debug import lcmt_sampling_c3_debug

class controller_subscriber():
    def __init__(self):    
        self.is_c3 = None
        self.lc = lcm.LCM()
        self.subscription = self.lc.subscribe("SAMPLING_C3_DEBUG", self.callback_)
    
    def callback_(self, channel, data):
        msg = lcmt_sampling_c3_debug.decode(data)
        self.is_c3 = msg.is_c3_mode

    def run(self):
        self.lc.handle_timeout(10)
    
    def get_controller_mode(self):
        return self.is_c3