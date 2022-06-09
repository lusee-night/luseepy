# these files are work in progress, but we should keep them
# as a track name conversions between Sim files and exported files

#export ROOT=$LUSEE_DRIVE_DIR/AntennaResponse/SimulationFiles/ShortDipoleComparison/dipole_in_vacuum/

export ROOT=$LUSEE_DRIVE_DIR/AntennaResponse/SimulationFiles/LuSEELanderRegolithComparison/eight_layer_regolith/ 
export OUTROOT=$LUSEE_DRIVE_DIR/Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith

#lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_1m_75deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_1m_75deg.fits
#lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_3m_75deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_3m_75deg.fits
#lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_6m_75deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_6m_75deg.fits

lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_1m_45deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_1m_45deg.fits
lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_3m_45deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_3m_45deg.fits
lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_6m_45deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_6m_45deg.fits

lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_1m_15deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_1m_15deg.fits
lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_3m_15deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_3m_15deg.fits
lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_6m_15deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_6m_15deg.fits




