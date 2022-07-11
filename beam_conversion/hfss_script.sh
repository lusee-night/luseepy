# these files are work in progress, but we should keep them
# as a track name conversions between Sim files and exported files

#export ROOT=$LUSEE_DRIVE_DIR/AntennaResponse/SimulationFiles/ShortDipoleComparison/dipole_in_vacuum/

export ROOT=$LUSEE_DRIVE_DIR/AntennaResponse/SimulationFiles/LuSEELanderFreeSpaceComparison/monopole_1m_75deg/hfss_lbl
export OUTROOT=$LUSEE_DRIVE_DIR/Simulations/BeamModels/LanderFreeSpaceComparison/

#python_dev beam_conversion/hfss.py -g --thetamax 181 $ROOT/dipole_Phase0deg -o $OUTROOT/hfss_lbl_1m_75.2portX.fits
#python_dev beam_conversion/hfss.py -g --thetamax 181 $ROOT/dipole_Phase180deg -o $OUTROOT/hfss_lbl_1m_75.2port.fits
#python_dev beam_conversion/hfss.py -g --thetamax 181 $ROOT/monopole_Phase0deg -o $OUTROOT/hfss_lbl_1m_75.fits



export ROOT=$LUSEE_DRIVE_DIR/AntennaResponse/SimulationFiles/LuSEELanderRegolithComparison/eight_layer_regolith/ 
export OUTROOT=$LUSEE_DRIVE_DIR/Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith

#lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_1m_75deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_1m_75deg.fits
#lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_1m_75deg/hfss_lbl/dipole_Phase180deg/ -o $OUTROOT/hfss_lbl_1m_75deg.2port.fits
#lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_1m_75deg/hfss_lbl/dipole_Phase0deg/ -o $OUTROOT/hfss_lbl_1m_75deg.2portX.fits


for L in 2m 3m
do
    for D in 75deg 45deg 15deg
    do
	for M in monopole_Phase0deg dipole_Phase180deg
	do
	    if [[ "$M" == *"dipole"* ]]; then
		export E=2port.fits
	    else
		export E=fits
	    fi
	    lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_$L\_$D/hfss_lbl/$M/ -o $OUTROOT/hfss_lbl_$L\_$D.$E
	done
  done
done
	 

#lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_6m_75deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_6m_75deg.fits

#lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_1m_45deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_1m_45deg.fits
#lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_3m_45deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_3m_45deg.fits
#lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_6m_45deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_6m_45deg.fits

#lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_1m_15deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_1m_15deg.fits
#lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_3m_15deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_3m_15deg.fits
#lpython_dev beam_conversion/hfss.py -g $ROOT/monopole_6m_15deg/hfss_lbl/monopole_Phase0deg/ -o $OUTROOT/hfss_lbl_6m_15deg.fits

