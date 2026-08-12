# Environment setup for a local (non-containerised) luseepy checkout.
#
# For your own environment, set these variables as shown in these examples:
#
# LUSEEPY_PATH="/home/user/work/lusee/luseepy"
# REFSPEC_PATH="/home/user/work/lusee/refspec"
# LUSEE_DRIVE_DIR="/home/user/work/lusee/Drive/"
# LUSEE_OUTPUT_DIR="/home/user/work/lusee/luseepy/simulation/output"

# load machine-specific overrides
if [ -f "$(dirname "${BASH_SOURCE[0]}")/setup_env.local.sh" ]; then
    source "$(dirname "${BASH_SOURCE[0]}")/setup_env.local.sh"
fi
