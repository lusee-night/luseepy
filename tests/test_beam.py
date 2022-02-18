import lusee

beam = lusee.LBeam('../AntennaSimResults/6m_out_of_phase.fits')
beam.rotate(90)
beam.flip_over_yz()
