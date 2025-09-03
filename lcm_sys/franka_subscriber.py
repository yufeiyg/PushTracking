import lcm
from lcm_sys.lcm_types.lcm_rb_pose import lcmt_sampling_c3_debug

class controller_subscriber():
    def __init__(self):    
        self.joint_positions = []
        self.lc = lcm.LCM()
        self.subscription = self.lc.subscribe("FRANKA_STATE", self.callback_)
    
    def callback_(self, channel, data):
        msg = lcmt_sampling_c3_debug.decode(data)
        self.joint_positions = msg.position

    def run(self):
        self.lc.handle_timeout(10)
    
    def get_joint_pos(self):
        return self.joint_positions