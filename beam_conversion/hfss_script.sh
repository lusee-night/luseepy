# these files are work in progress, but we should keep them
# as a track name conversions between Sim files and exported files

#export ROOT=$LUSEE_DRIVE_DIR/AntennaResponse/SimulationFiles/ShortDipoleComparison/dipole_in_vacuum/

export ROOT=$LUSEE_DRIVE_DIR/AntennaResponse/SimulationFiles/LuSEELanderRegolithComparison/eight_layer_regolith/ 
export OUTROOT=$LUSEE_DRIVE_DIR/Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith

lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_1m_75deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_1m_75deg.fits
lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_1m_75deg/hfss_lbl/dipole_Phase180deg/ -o $OUTROOT/hfss_lbl_1m_75deg.2port.fits
lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_1m_15deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_1m_15deg.fits
lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_1m_15deg/hfss_lbl/dipole_Phase180deg/ -o $OUTROOT/hfss_lbl_1m_15deg.2port.fits


