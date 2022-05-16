# these files are work in progress, but we should keep them
# as a track name conversions between Sim files and exported files

#export ROOT=$LUSEE_DRIVE_DIR/AntennaResponse/SimulationFiles/ShortDipoleComparison/dipole_in_vacuum/

export ROOT=$LUSEE_DRIVE_DIR/AntennaResponse/SimulationFiles/LuSEELanderRegolithComparison/eight_layer_regolith/ 
export OUTROOT=$LUSEE_DRIVE_DIR/AntennaResponse/Exported/LuSEELanderRegolithComparison/eight_layer_regolith/
lpython_dev beam_conversion/hfss.py $ROOT/monopole_1m_75deg/hfss_lbl/ -o $OUTROOT/lbl_monopole_1m_75deg.fits
#lpython_dev beam_conversion/hfss.py $ROOT/monopole_3m_75deg/hfss_lbl/ -o $OUTROOT/lbl_monopole_3m_75deg.fits

