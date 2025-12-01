class Trial:
    
    def __init__(self, name, session, date, number, odor_name, session_type, delay_time, go_time, refract_time, 
                 pulse_time, high_count, low_count, reward_side, lick_side, odor_profile, 
                 lick0_profile, lick1_profile, breath_profile, pre_breath_profile):
        self.name = name
        self.date = date
        self.session = session 
        self.number = number
        self.odor_name = odor_name
        self.session_type = session_type
        self.delay_time = delay_time
        self.go_time = go_time
        self.refract_time = refract_time
        self.pulse_time = pulse_time
        self.high_count = high_count
        self.low_count = low_count
        self.reward_side = reward_side
        self.lick_side = lick_side
        self.odor_profile = odor_profile
        self.lick0_profile = lick0_profile
        self.lick1_profile = lick1_profile
        self.breath_profile = breath_profile
        self.pre_breath_profile = pre_breath_profile
        
        self.num_pulses = self.odor_profile.sum()/100/pulse_time
        self.correct = self.reward_side == self.lick_side
        
    def __str__(self):
        return '{} trial of {}, session {}, number {}'.format(self.session_type, self.name, self.session, self.number)
    
    def __repr__(self):
        return 'Trial({},{},{},{})'.format(self.session_type, self.name, self.session, self.number)
