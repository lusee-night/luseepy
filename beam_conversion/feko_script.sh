# these files are work in progress, but we should keep them
# as a track name conversions between Sim files and exported files

export ROOT=$LUSEE_DRIVE_DIR/AntennaResponse/SimulationFiles/ShortDipoleComparison/dipole_in_vacuum/feko_bnl/20220406_short_dipole_5mm
export OUT=$LUSEE_DRIVE_DIR/Simulations/BeamModels/ShortDipoleComparison/dipole_in_vacuum/feko_bnl_short_dipole_5mm.2port.fits
#lpython_dev beam_conversion/feko.py $ROOT -o $OUT


export ROOT=$LUSEE_DRIVE_DIR/AntennaResponse/SimulationFiles/LuSEELanderRegolithComparison/single_layer_epsilon3pt7_tan0pt01
export OUTROOT=$LUSEE_DRIVE_DIR//Simulations/BeamModels/LanderRegolithComparison/single_layer_epsilon3pt7_tan0pt01

#lpython_dev beam_conversion/feko.py $ROOT/monopole_1m_75deg/feko_bnl/20220419_v1_cylinder_lander -o $OUTROOT/feko_bnl_1m_75deg_cyl.2port.fits

export ROOT=$LUSEE_DRIVE_DIR/AntennaResponse/SimulationFiles/LuSEELanderRegolithComparison/eight_layer_regolith
export OUTROOT=$LUSEE_DRIVE_DIR//Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith

lpython_dev beam_conversion/feko.py $ROOT/monopole_1m_15deg/feko_bnl/20220517_v5pt12_1m_15deg_2_port_1MHz -o $OUTROOT/feko_bnl_1m_15deg.2port.fits
lpython_dev beam_conversion/feko.py $ROOT/monopole_1m_15deg/feko_bnl/20220518_v5pt15_1m_15deg_1_port_1MHz -o $OUTROOT/feko_bnl_1m_15deg.fits
lpython_dev beam_conversion/feko.py $ROOT/monopole_1m_45deg/feko_bnl/20220517_v5pt13_1m_45deg_2_port_1MHz -o $OUTROOT/feko_bnl_1m_45deg.2port.fits
lpython_dev beam_conversion/feko.py $ROOT/monopole_1m_45deg/feko_bnl/20220518_v5pt15_1m_45deg_1_port_1MHz -o $OUTROOT/feko_bnl_1m_45deg.fits
lpython_dev beam_conversion/feko.py $ROOT/monopole_1m_75deg/feko_bnl/20220517_v5pt14_1m_75deg_2_port_1MHz -o $OUTROOT/feko_bnl_1m_75deg.2port.fits
lpython_dev beam_conversion/feko.py $ROOT/monopole_1m_75deg/feko_bnl/20220518_v5pt17_1m_75deg_1_port_1MHz -o $OUTROOT/feko_bnl_1m_75deg.fits






#export ROOT=$LUSEE_DRIVE_DIR/AntennaResponse/SimulationFiles/LuSEELanderRegolithComparison/eight_layer_regolith
#export OUTROOT=$LUSEE_DRIVE_DIR/AntennaResponse/Exported/LuSEELanderRegolithComparison/eight_layer_regolith/ 

#lpython_dev beam_conversion/feko.py $ROOT/monopole_1m_75deg/feko_bnl/20220427_v5_complex_soil/LuSEE_moon_v5_complex_soil -o $OUTROOT/feko_monopole_1m_75deg.fits
#lpython_dev beam_conversion/feko.py $ROOT/monopole_1m_45deg/feko_bnl/20220429_v5pt1_1m_45deg/LuSEE_moon_v5pt1_45deg -o $OUTROOT/feko_monopole_1m_45deg.fits
#lpython_dev beam_conversion/feko.py $ROOT/monopole_1m_15deg/feko_bnl/20220429_v5pt2_1m_15deg/LuSEE_moon_v5pt2_15deg -o $OUTROOT/feko_monopole_1m_15deg.fits
#
#lpython_dev beam_conversion/feko.py $ROOT/monopole_3m_15deg/feko_bnl/20220509_v5pt8_3m_15deg/LuSEE_moon_v5pt8_3m_15deg -o $OUTROOT/feko_monopole_3m_15deg.fits
#lpython_dev beam_conversion/feko.py $ROOT/monopole_3m_45deg/feko_bnl/20220505_v5pt7_3m_45deg/LuSEE_moon_v5pt7_3m_45deg -o $OUTROOT/feko_monopole_3m_45deg.fits
#lpython_dev beam_conversion/feko.py $ROOT/monopole_3m_75deg/feko_bnl/20220504_v5pt6_3m_75deg/LuSEE_moon_v5pt6_3m_75deg -o $OUTROOT/feko_monopole_3m_75deg.fits
#lpython_dev beam_conversion/feko.py $ROOT/monopole_6m_15deg/feko_bnl/20220510_v5pt5_6m_15deg/LuSEE_moon_v5pt5_6m_15deg -o $OUTROOT/feko_monopole_6m_15deg.fits
#lpython_dev beam_conversion/feko.py $ROOT/monopole_6m_45deg/feko_bnl/20220501_v5pt4_6m_45deg/LuSEE_moon_v5pt4_6m_45deg -o $OUTROOT/feko_monopole_6m_45deg.fits
#lpython_dev beam_conversion/feko.py $ROOT/monopole_6m_75deg/feko_bnl/20220510_v5pt3_6m_75deg/LuSEE_moon_v5pt3_6m_75deg -o $OUTROOT/feko_monopole_6m_75deg.fits
