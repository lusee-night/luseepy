#!/usr/bin/env python
#
# This scripts can be used to create pdr_config.batch

antlist = "hfss_lbl_2p5m_75deg hfss_lbl_2p7mAntennaHeight_1p5mLanderHeight hfss_lbl_3p2mAntennaHeight_2p0mLanderHeight_Baseline hfss_lbl_3p7mAntennaHeight_2p5mLanderHeight hfss_lbl_4p2mAntennaHeight_3p0mLanderHeight hfss_lbl_4p7mAntennaHeight_2p0mLanderHeight_2mLanderThickness hfss_lbl_4p7mAntennaHeight_3p5mLanderHeight hfss_lbl_5p2mAntennaHeight_4p0mLanderHeight hfss_lbl_5p7mAntennaHeight_4p5mLanderHeight".split()
antlist = "hfss_lbl_2m_75deg hfss_lbl_3m_75deg".split()
for ant in antlist:
    for bangle in range(10,95,10):
        print (f"observation:dt=60 beam_config:common_beam_angle={bangle} beam_config:default_file={ant}.fits beam_config:couplings:opposite:two_port={ant}.2port.fits simulation:output={ant}_R{bangle}.fits simulation:cache_transform=lunar_day_pdr_{bangle}.pickle")
        if bangle<15:
            print (f"sky:type=CMB observation:dt=3600*24*4 beam_config:common_beam_angle={bangle} beam_config:default_file={ant}.fits beam_config:couplings:opposite:two_port={ant}.2port.fits simulation:output={ant}_R{bangle}_CMB.fits")
            print (f"sky:type=DarkAges sky:nu_rms=14 sky:scaled=1 observation:dt=3600*24*4 beam_config:common_beam_angle={bangle} beam_config:default_file={ant}.fits beam_config:couplings:opposite:two_port={ant}.2port.fits simulation:output={ant}_R{bangle}_14MHz_DA.fits")


        
        
    


